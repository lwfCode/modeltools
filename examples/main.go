package main

import (
	"fmt"
	"github.com/lwfCode/modeltools"
)

type MyResult struct {
	AlertType  string  `json:"alert_type"`
	Scene      string  `json:"scene"`
	Risk       string  `json:"risk"`
	Suggestion string  `json:"suggestion"`
	Confidence float64 `json:"confidence"`
	Level      string  `json:"level"`
}

func main() {
	prompt := `
你是校园安全分析AI，请根据图片内容作出判断。
告警类型枚举（alert_type 只能取以下值之一）：
- 校门关闭       （校门关闭）
- 校门开启       （校门开启）
- 校园欺凌     （打架、推搡、围堵学生）
- 人群聚集     （楼梯/走廊拥堵、踩踏风险）
- 危险行为     （翻越围栏、攀爬、高空危险动作）
- 可疑人员     （陌生人非法进入校园）
- 消防隐患     （消防通道堵塞、灭火器缺失）
- 学生倒地不起     （学生倒地不起，需要人工干预）
- 电梯内抽烟     （电梯抽烟行为）
- 无风险       （未发现安全隐患）
严格只返回如下 JSON，不要包裹 markdown，不要多余内容：

{
  "alert_type": "以上枚举之一",
  "scene": "用一句话描述当前画面场景",
  "risk": "具体风险描述，无风险时填'未发现安全风险'",
  "suggestion": "简短处置建议，无风险时填'继续监控'",
  "confidence": 0.00,
  "level": "低/中/高"
}
`

	res, err := modeltools.RunWith[MyResult](
		modeltools.Config{
			APIKey:       "you api key",
			BaseURL:      "you base url",
			Model:        "you model name",
			SystemPrompt: "你是一个专业的校园安全分析助手。",
			TimeoutSec:   120,
		},
		prompt,
		"/path/to/image.jpg",
		true, // 开启终端打印
	)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("Result: %+v\n", res)
}

