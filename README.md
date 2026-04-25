# lwfmodel

通用多模态大语言模型调用包，支持传入**提示词 + 图片**，将 AI 返回的 JSON 自动反序列化到你自定义的结构体。

## 安装

```bash
go get github.com/leiwenfeng/lwfmodel
```

## 环境变量配置

| 变量名                | 说明                                         | 默认值                        |
| --------------------- | -------------------------------------------- | ----------------------------- |
| `MOONSHOT_API_KEY`    | API 鉴权密钥（必填）                         | —                             |
| `MOONSHOT_BASE_URL`   | API 地址（兼容任意 OpenAI 格式接口）         | `https://api.moonshot.cn/v1`  |
| `MOONSHOT_MODEL`      | 模型名称                                     | `kimi-k2.5`                   |
| `MOONSHOT_TIMEOUT_SEC`| 请求超时秒数                                 | `120`                         |
| `MOONSHOT_MAX_TOKENS` | 最大回复 token 数（0 = 不限制）              | —                             |

## 快速开始

### 1. 纯文本问答

```go
package main

import (
    "fmt"
    "github.com/leiwenfeng/lwfmodel"
)

type TranslateResult struct {
    Result   string `json:"result"`
    Language string `json:"language"`
}

func main() {
    prompt := `把下面的文字翻译成英文，只返回 JSON：
{"result": "翻译结果", "language": "目标语言"}

原文：你好，世界`

    res, err := lwfmodel.Run[TranslateResult](prompt, "", true)
    if err != nil {
        panic(err)
    }
    fmt.Println(res.Result)
}
```

### 2. 图片分析（本地文件 / URL / base64 均支持）

```go
type ImageAnalysis struct {
    Scene      string  `json:"scene"`
    HasPeople  bool    `json:"has_people"`
    Confidence float64 `json:"confidence"`
}

func main() {
    prompt := `分析这张图片，只返回 JSON（不加 markdown）：
{
  "scene": "一句话描述画面",
  "has_people": true,
  "confidence": 0.95
}`

    // 支持三种图片来源：
    // res, err := lwfmodel.Run[ImageAnalysis](prompt, "/path/to/local.jpg", true)
    // res, err := lwfmodel.Run[ImageAnalysis](prompt, "https://example.com/img.jpg", true)
    res, err := lwfmodel.Run[ImageAnalysis](prompt, "/path/to/image.jpg", true)
    if err != nil {
        panic(err)
    }
    fmt.Printf("场景: %s, 有人: %v, 置信度: %.2f\n", res.Scene, res.HasPeople, res.Confidence)
}
```

### 3. 自定义配置（不依赖环境变量）

```go
res, err := lwfmodel.RunWith[ImageAnalysis](
    lwfmodel.Config{
        APIKey:       "sk-xxxxxxxxxxxxxxxx",
        BaseURL:      "https://api.moonshot.cn/v1",
        Model:        "kimi-k2.5",
        SystemPrompt: "你是一个专业的图像分析助手。",
        TimeoutSec:   60,
    },
    prompt,
    "/path/to/image.jpg",
    true, // 开启终端打印
)
```

### 4. 接入本地模型（Ollama 等）

```go
res, err := lwfmodel.RunWith[MyResult](
    lwfmodel.Config{
        APIKey:    "ollama",  // 本地模型一般不需要真实 key
        BaseURL:   "http://localhost:11434/v1",
        Model:     "llava",
        MaxTokens: 512,
    },
    prompt, image, false,
)
```

## API 文档

### `Run[T any](prompt, image string, enablePrint bool) (*T, error)`

使用环境变量配置调用 AI，返回解析后的自定义结构体。

| 参数          | 类型     | 说明                                                              |
| ------------- | -------- | ----------------------------------------------------------------- |
| `prompt`      | `string` | 提示词，建议在末尾说明期望的 JSON 格式                            |
| `image`       | `string` | 图片来源：本地路径 / http(s) URL / base64 data URL，`""` 为纯文本 |
| `enablePrint` | `bool`   | 是否在终端打印彩色结果摘要                                        |

### `RunWith[T any](cfg Config, prompt, image string, enablePrint bool) (*T, error)`

同 `Run`，但允许显式传入 `Config` 结构体覆盖环境变量。

### `Config` 结构体

```go
type Config struct {
    APIKey       string // API 密钥
    BaseURL      string // API 地址
    Model        string // 模型名称
    SystemPrompt string // 系统消息（可选）
    TimeoutSec   int    // 超时秒数
    MaxTokens    int64  // 最大回复 token 数
}
```

## 注意事项

- **JSON 格式**：提示词中须明确告知 AI 只返回 JSON，不要包裹 markdown。
- **图片压缩**：包内会自动将图片压缩到 768px 以内再发送，减少 token 消耗。
- **泛型支持**：需要 Go 1.21+。
- **模型兼容**：BaseURL 兼容任意 OpenAI 格式接口（Kimi / DeepSeek / Ollama / vLLM 等）。
