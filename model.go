// Package modeltools 是一个通用的多模态大语言模型调用包。
// 支持传入提示词、图片（本地路径 / URL / base64），将 AI 返回的 JSON 自动反序列化到
// 调用方自定义的结构体，并可选开启彩色终端打印。
//
// 使用示例：
//
//	type MyResult struct {
//	    Summary string  `json:"summary"`
//	    Score   float64 `json:"score"`
//	}
//
//	result, err := modeltools.Run[MyResult](
//	    "请分析这张图片并以 JSON 返回：{\"summary\":\"...\"\"score\":0.0}",
//	    "/path/to/image.jpg",
//	    true, // 开启终端打印
//	)
package modeltools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Config 大模型配置。所有字段均可选，未填写时自动读取对应环境变量。
type Config struct {
	// APIKey 鉴权密钥。环境变量：MOONSHOT_API_KEY
	APIKey string
	// BaseURL API 地址。环境变量：MOONSHOT_BASE_URL（默认 https://api.moonshot.cn/v1）
	BaseURL string
	// Model 模型名称。环境变量：MOONSHOT_MODEL（默认 kimi-k2.5）
	Model string
	// SystemPrompt 系统消息（可选）。留空则不发送 system 消息。
	SystemPrompt string
	// TimeoutSec 请求超时秒数。环境变量：MOONSHOT_TIMEOUT_SEC（默认 120）
	TimeoutSec int
	// MaxTokens 限制回复 token 数（0 表示不限制）。环境变量：MOONSHOT_MAX_TOKENS
	MaxTokens int64
}

// resolve 填充空字段的默认值（优先使用显式配置，其次读环境变量，最后取内置默认值）
func (c Config) resolve() Config {
	if c.APIKey == "" {
		c.APIKey = os.Getenv("MOONSHOT_API_KEY")
	}
	if c.BaseURL == "" {
		c.BaseURL = os.Getenv("MOONSHOT_BASE_URL")
	}
	if c.BaseURL == "" {
		c.BaseURL = "https://api.moonshot.cn/v1"
	}
	if c.Model == "" {
		c.Model = os.Getenv("MOONSHOT_MODEL")
	}
	if c.Model == "" {
		c.Model = "kimi-k2.5"
	}
	if c.TimeoutSec <= 0 {
		fmt.Sscan(os.Getenv("MOONSHOT_TIMEOUT_SEC"), &c.TimeoutSec) //nolint:errcheck
		if c.TimeoutSec <= 0 {
			c.TimeoutSec = 120
		}
	}
	if c.MaxTokens == 0 {
		fmt.Sscan(os.Getenv("MOONSHOT_MAX_TOKENS"), &c.MaxTokens) //nolint:errcheck
	}
	return c
}

// Run 使用环境变量中的配置调用大模型。
//
//   - prompt: 用户提示词（应包含期望的 JSON 格式说明）
//   - image:  图片来源，支持本地路径 / http(s) URL / base64 data URL，传 "" 表示纯文本请求
//   - enablePrint: 是否在终端打印彩色结果摘要
//
// T 为调用方自定义的返回结构体，字段名须与 AI 返回的 JSON key 匹配。
func Run[T any](prompt, image string, enablePrint bool) (*T, error) {
	return RunWith[T](Config{}, prompt, image, enablePrint)
}

// RunWith 同 Run，但允许传入显式 Config 覆盖环境变量。
func RunWith[T any](cfg Config, prompt, image string, enablePrint bool) (*T, error) {
	cfg = cfg.resolve()
	start := time.Now()

	if enablePrint {
		fmt.Printf("\033[33m⏳ 分析开始 %s ...\033[0m\n", start.Format("15:04:05"))
	}

	result, raw, err := callLLM[T](cfg, prompt, image)
	if err != nil {
		return nil, err
	}

	if enablePrint {
		elapsed := time.Since(start).Round(time.Millisecond)
		fmt.Printf("\033[32m✓ 分析完成 %s (耗时 %s)\033[0m\n",
			time.Now().Format("15:04:05"), elapsed)
		printResult(raw)
	}

	return result, nil
}

// callLLM 负责构造请求、调用 API、提取 JSON、反序列化到 T。
func callLLM[T any](cfg Config, prompt, imageInput string) (*T, string, error) {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(cfg.BaseURL),
	)

	ctx, cancel := context.WithTimeout(context.Background(),
		time.Duration(cfg.TimeoutSec)*time.Second)
	defer cancel()

	// 构造用户消息内容（图片 + 文字）
	var userParts []openai.ChatCompletionContentPartUnionParam
	if imageInput != "" {
		imageURL, err := resolveImageURL(imageInput)
		if err != nil {
			return nil, "", fmt.Errorf("处理图片失败: %w", err)
		}
		userParts = append(userParts, openai.ImageContentPart(
			openai.ChatCompletionContentPartImageImageURLParam{URL: imageURL},
		))
	}
	userParts = append(userParts, openai.TextContentPart(prompt))

	// 构造消息列表
	var messages []openai.ChatCompletionMessageParamUnion
	if cfg.SystemPrompt != "" {
		messages = append(messages, openai.SystemMessage(cfg.SystemPrompt))
	}
	messages = append(messages, openai.UserMessage(userParts))

	params := openai.ChatCompletionNewParams{
		Model:    cfg.Model,
		Messages: messages,
	}

	if cfg.MaxTokens > 0 {
		if cfg.MaxTokens > 8192 {
			cfg.MaxTokens = 8192
		}
		params.MaxTokens = openai.Int(cfg.MaxTokens)
	}

	resp, err := client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, "", fmt.Errorf("调用 AI 接口失败: %w", err)
	}

	raw := extractJSON(resp.Choices[0].Message.Content)

	var result T
	if err := json.NewDecoder(strings.NewReader(raw)).Decode(&result); err != nil {
		return nil, raw, fmt.Errorf("解析 AI 返回 JSON 失败: %w\n原始内容: %s", err, raw)
	}

	return &result, raw, nil
}

// printResult 将 AI 返回的 JSON 美化后打印到终端。
func printResult(raw string) {
	bar := strings.Repeat("─", 52)
	blue := "\033[1;36m"
	yellow := "\033[1;33m"
	red := "\033[1;31m"
	green := "\033[1;32m"
	reset := "\033[0m"

	fmt.Printf("\n%s┌%s┐%s\n", blue, bar, reset)
	fmt.Printf("%s│  🤖 AI 返回结果%-28s│%s\n", blue, "", reset)
	fmt.Printf("%s├%s┤%s\n", blue, bar, reset)

	var m map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &m); err == nil {
		for _, k := range []string{"scene", "risk", "suggestion", "confidence", "level", "alert_type"} {
			if v, ok := m[k]; ok {
				printField(k, v, yellow, blue, red, green, reset)
				delete(m, k)
			}
		}

		if len(m) > 0 {
			keys := make([]string, 0, len(m))
			for k := range m {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				printField(k, m[k], yellow, blue, red, green, reset)
			}
		}
	} else {
		fmt.Printf("%s│%s  %s\n", blue, reset, raw)
	}

	fmt.Printf("%s└%s┘%s\n\n", blue, bar, reset)

	fmt.Printf("%sJSON 原始结果:%s\n", blue, reset)
	printJSON(raw, yellow, reset)
}

func printField(key string, value interface{}, yellow, blue, red, green, reset string) {
	val := fmt.Sprintf("%v", value)
	runes := []rune(val)
	if len(runes) > 38 {
		val = string(runes[:35]) + "..."
	}

	color := blue
	if key == "level" {
		lower := strings.ToLower(strings.TrimSpace(val))
		switch lower {
		case "高", "high":
			color = red
		case "中", "middle", "medium":
			color = yellow
		case "低", "low":
			color = green
		default:
			color = yellow
		}
	}

	fmt.Printf("%s│%s  %s%-14s%s %s%-36s%s%s\n",
		yellow, reset, yellow, key+":", reset, color, val, reset, blue)
}

func printJSON(raw, yellow, reset string) {
	var m map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &m); err != nil {
		fmt.Println(raw)
		return
	}

	fmt.Printf("%s{\n", yellow)
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for i, k := range keys {
		v := m[k]
		val := fmt.Sprintf("%v", v)
		if _, ok := v.(string); ok {
			val = fmt.Sprintf("\"%s\"", val)
		}
		comma := ","
		if i == len(keys)-1 {
			comma = ""
		}
		fmt.Printf("  \"%s%s%s\": %s%s%s%s\n",
			yellow, k, reset, reset, val, reset, comma)
	}
	fmt.Printf("}%s\n", reset)
}

// extractJSON 从 AI 原始输出中提取第一个完整的 JSON 对象。
// 处理：markdown 代码块包裹（```json ... ```）和 <think>...</think> 推理链。
func extractJSON(raw string) string {
	s := strings.TrimSpace(raw)

	// 去除 <think>...</think> 推理链（DeepSeek-R1 等模型会输出）
	for {
		ts := strings.Index(s, "<think>")
		if ts == -1 {
			break
		}
		te := strings.Index(s, "</think>")
		if te == -1 || te <= ts {
			s = strings.TrimSpace(s[:ts])
			break
		}
		s = strings.TrimSpace(s[:ts] + s[te+len("</think>"):])
	}

	// 去掉 markdown 代码围栏
	for _, fence := range []string{"```json", "```"} {
		if strings.HasPrefix(s, fence) {
			s = strings.TrimPrefix(s, fence)
			if idx := strings.Index(s, "```"); idx != -1 {
				s = s[:idx]
			}
			s = strings.TrimSpace(s)
			break
		}
	}

	// 括号配对找第一个完整 JSON 对象
	depth, start, inStr, escaped := 0, -1, false, false
	for i, ch := range s {
		switch {
		case escaped:
			escaped = false
		case ch == '\\' && inStr:
			escaped = true
		case ch == '"':
			inStr = !inStr
		case !inStr && ch == '{':
			if depth == 0 {
				start = i
			}
			depth++
		case !inStr && ch == '}':
			depth--
			if depth == 0 && start != -1 {
				return strings.TrimSpace(s[start : i+1])
			}
		}
	}

	return strings.TrimSpace(s)
}
