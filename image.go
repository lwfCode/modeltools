package lwfmodel

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	_ "image/gif"  // 注册 gif 解码器
	_ "image/png"  // 注册 png 解码器
	"io"
	"net/http"
	"os"
	"strings"
)

// resolveImageURL 将三种图片来源统一转换为 base64 data URL：
//   - 已经是 data URL（"data:image/..."）→ 解码后重新压缩再编码
//   - http/https URL                      → 下载后转 base64
//   - 本地文件路径                         → 读取后转 base64
func resolveImageURL(input string) (string, error) {
	switch {
	case strings.HasPrefix(input, "data:image/"):
		semi := strings.Index(input, ";base64,")
		if semi == -1 {
			return input, nil
		}
		b64Data := input[semi+len(";base64,"):]
		imgBytes, err := base64.StdEncoding.DecodeString(b64Data)
		if err != nil {
			return input, nil // 解码失败则原样返回
		}
		compressed := compressImageBytes(imgBytes, 768)
		encoded := base64.StdEncoding.EncodeToString(compressed)
		return "data:image/jpeg;base64," + encoded, nil

	case strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://"):
		resp, err := http.Get(input) //nolint:noctx
		if err != nil {
			return "", fmt.Errorf("下载图片失败: %w", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return "", fmt.Errorf("下载图片失败，状态码: %d", resp.StatusCode)
		}
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("读取图片数据失败: %w", err)
		}
		data = compressImageBytes(data, 768)
		return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data), nil

	default:
		// 本地文件路径
		data, err := os.ReadFile(input)
		if err != nil {
			return "", fmt.Errorf("读取图片文件失败: %w", err)
		}
		data = compressImageBytes(data, 768)
		return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data), nil
	}
}

// compressImageBytes 将图片压缩到 maxDim×maxDim 以内，以 JPEG 75% 质量重新编码。
// 可大幅降低 base64 payload 大小，减少模型 token 消耗。
// 若解码失败则原样返回，避免因压缩错误阻断正常流程。
func compressImageBytes(data []byte, maxDim int) []byte {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return data
	}

	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	if w > maxDim || h > maxDim {
		var newW, newH int
		if w > h {
			newW, newH = maxDim, h*maxDim/w
		} else {
			newW, newH = w*maxDim/h, maxDim
		}
		resized := image.NewRGBA(image.Rect(0, 0, newW, newH))
		for y := 0; y < newH; y++ {
			for x := 0; x < newW; x++ {
				resized.Set(x, y, img.At(x*w/newW, y*h/newH))
			}
		}
		img = resized
	}

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 75}); err != nil {
		return data
	}
	return buf.Bytes()
}
