# 期货三重滤网交易系统

基于 TQSDK（天勤SDK）实现的期货三重滤网交易系统。

## 三重滤网策略原理

### 第一层滤网 - 趋势判断
使用 EMA（指数移动平均线）判断市场方向：
- **长期 EMA**（日线 26 周期）
- 价格在 EMA 上方 = 多头趋势 📈
- 价格在 EMA 下方 = 空头趋势 📉

### 第二层滤网 - 振荡指标
使用 RSI（相对强弱指标）：
- RSI > 70 = 超买区域（做空信号）🔴
- RSI < 30 = 超卖区域（做多信号）🟢
- 结合趋势方向过滤信号

### 第三层滤网 - 入场点确认
使用价格通道和 ATR（平均真实波幅）：
- **多头**：等待回调到支撑位（下轨），RSI 回升后入场
- **空头**：等待反弹到压力位（上轨），RSI 回落后期权做空

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 完整系统模式（需要天勤终端在线）
```bash
python triple_screen_trading.py
```

### 指定交易品种
```bash
python triple_screen_trading.py --symbol DCE.j2501
```

### 演示模式（无需连接）
```bash
python triple_screen_trading.py --demo
```

### 调试模式
```bash
python triple_screen_trading.py --debug
```

### 自定义运行时间
```bash
python triple_screen_trading.py --duration 300  # 运行5分钟
```

## 核心模块

### TechnicalIndicators
技术指标计算类，提供：
- `calculate_ema()` - EMA 指数移动平均线
- `calculate_rsi()` - RSI 相对强弱指标
- `calculate_atr()` - ATR 平均真实波幅
- `calculate_price_channel()` - 价格通道

### TripleScreenStrategy
三重滤网策略引擎：
- 趋势判断（第一层）
- RSI 振荡分析（第二层）
- 入场点确认（第三层）
- 交易信号生成

### TripleScreenTradingSystem
交易系统主类：
- TQSDK 连接管理
- 实时行情订阅
- 多品种监控
- 信号面板展示

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| EMA_SHORT | 12 | 短期 EMA 周期 |
| EMA_LONG | 26 | 长期 EMA 周期 |
| RSI_PERIOD | 14 | RSI 周期 |
| RSI_OVERBOUGHT | 70 | RSI 超买阈值 |
| RSI_OVERSOLD | 30 | RSI 超卖阈值 |
| ATR_PERIOD | 14 | ATR 周期 |
| STOP_LOSS_ATR_MULTI | 2.0 | 止损 ATR 倍数 |
| TAKE_PROFIT_ATR_MULTI | 3.0 | 止盈 ATR 倍数 |

## 默认监控品种

- DCE.j2505 - 焦煤期货（当前主力）
- DCE.m2505 - 豆粕期货
- DCE.y2505 - 豆油期货
- CZCE.RM405 - 菜粕期货
- CZCE.OI405 - 菜油期货

## 配置

### 方式一：.env 文件（推荐）

```bash
# 1. 复制配置文件
cp .env.example .env

# 2. 编辑 .env 文件，填入你的账号密码
nano .env

# 3. 运行
python triple_screen_trading.py
```

### 方式二：环境变量

```bash
# Linux/Mac
export TQ_ACCOUNT='你的天勤账号'
export TQ_PASSWORD='你的天勤密码'
python triple_screen_trading.py

# Windows
set TQ_ACCOUNT=你的天勤账号
set TQ_PASSWORD=你的天勤密码
python triple_screen_trading.py
```

> ⚠️ 重要：`.env` 文件已加入 `.gitignore`，不会提交到 GitHub！

## 注意事项

1. **环境变量**：必须设置 `TQ_ACCOUNT` 和 `TQ_PASSWORD` 环境变量
2. **网络要求**：需要天勤终端在线才能获取实时数据
3. **风险提示**：本系统仅供学习研究，实盘交易请谨慎
4. **数据延迟**：注意行情数据的延迟情况

## 作者

寇豆码量化团队

## 版本

v1.0
