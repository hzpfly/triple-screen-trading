#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
期货三重滤网交易系统
基于 TQSDK 实现三重滤网策略

三重滤网原理：
- 第一层：趋势 EMA（判断方向）
- 第二层：RSI 振荡指标（判断超买超卖）
- 第三层：价格通道（确定入场点）

作者：寇豆码量化团队
版本：v1.0
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

# TQSDK 核心库
from tqsdk import TqApi, TqAuth, TargetPosTask

# 尝试导入可选依赖
try:
    from dotenv import load_dotenv
    # 优先从 .env 文件加载配置
    load_dotenv()
except ImportError:
    print("[警告] python-dotenv 未安装，将仅使用环境变量")

# 尝试导入可选依赖
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("[警告] numpy 未安装，将使用纯 Python 实现指标计算")

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("[警告] tabulate 未安装，将使用简化输出")


# =============================================================================
# 配置参数
# =============================================================================

# 天勤账号配置（从 .env 文件或环境变量读取）
TQ_ACCOUNT = os.getenv("TQ_ACCOUNT", "")
TQ_PASSWORD = os.getenv("TQ_PASSWORD", "")

# 如果仍未设置，尝试从 .env 文件读取（兼容无 dotenv 的环境）
if not TQ_ACCOUNT or not TQ_PASSWORD:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        if key == "TQ_ACCOUNT" and not TQ_ACCOUNT:
                            TQ_ACCOUNT = value.strip()
                        elif key == "TQ_PASSWORD" and not TQ_PASSWORD:
                            TQ_PASSWORD = value.strip()

# 验证账号和密码必须设置
if not TQ_ACCOUNT:
    raise ValueError(
        "❌ 错误：未设置 TQ_ACCOUNT\n"
        "   请选择以下方式之一配置：\n"
        "   1. 创建 .env 文件（参考 .env.example）\n"
        "   2. 设置环境变量：export TQ_ACCOUNT='你的账号'"
    )

if not TQ_PASSWORD:
    raise ValueError(
        "❌ 错误：未设置 TQ_PASSWORD\n"
        "   请选择以下方式之一配置：\n"
        "   1. 创建 .env 文件（参考 .env.example）\n"
        "   2. 设置环境变量：export TQ_PASSWORD='你的密码'"
    )

# 技术指标参数 - 三重滤网核心参数
EMA_SHORT = 12          # 短期 EMA 周期
EMA_LONG = 26           # 长期 EMA 周期
RSI_PERIOD = 14         # RSI 周期
RSI_OVERBOUGHT = 70     # RSI 超买阈值
RSI_OVERSOLD = 30       # RSI 超卖阈值
ATR_PERIOD = 14         # ATR 周期

# 交易参数
STOP_LOSS_ATR_MULTI = 2.0    # 止损 ATR 倍数
TAKE_PROFIT_ATR_MULTI = 3.0  # 止盈 ATR 倍数

# 默认交易品种（支持自定义）- 使用当前主力合约
DEFAULT_SYMBOLS = [
    "DCE.j2505",    # 焦煤期货（当前主力）
    "DCE.m2505",    # 豆粕期货
    "DCE.y2505",    # 豆油期货
    "CZCE.RM405",   # 菜粕期货
    "CZCE.OI405",   # 菜油期货
]

# 时间框架配置
DAILY_KLINE = (1, 86400)      # 日线
HOURLY_KLINE = (1, 3600)      # 小时线
MINUTE_5_KLINE = (5, 60)     # 5分钟线


# =============================================================================
# 技术指标计算模块
# =============================================================================

class TechnicalIndicators:
    """技术指标计算类"""
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> Optional[float]:
        """
        计算指数移动平均线 (EMA)
        
        Args:
            prices: 价格序列
            period: EMA 周期
            
        Returns:
            最新 EMA 值，数据不足返回 None
        """
        if not prices or len(prices) < period:
            return None
            
        if NUMPY_AVAILABLE:
            # 使用 numpy 的 ewm 方法计算 EMA
            prices_array = np.array(prices)
            ema = prices_array[-1]  # 初始化为最新价格
            alpha = 2.0 / (period + 1)
            # 使用 pandas 风格的 ewm 计算
            ema_values = []
            for i in range(len(prices_array)):
                if i == 0:
                    ema_values.append(prices_array[i])
                else:
                    ema_values.append(alpha * prices_array[i] + (1 - alpha) * ema_values[-1])
            return ema_values[-1]
        else:
            # 纯 Python 实现
            alpha = 2.0 / (period + 1)
            ema = sum(prices[:period]) / period  # SMA 作为初始值
            
            for price in prices[period:]:
                ema = alpha * price + (1 - alpha) * ema
                
            return ema
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """
        计算相对强弱指标 (RSI)
        
        Args:
            prices: 价格序列（收盘价）
            period: RSI 周期
            
        Returns:
            最新 RSI 值 (0-100)，数据不足返回 None
        """
        if not prices or len(prices) < period + 1:
            return None
            
        # 计算价格变动
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        # 计算平均收益和平均损失
        if len(gains) < period:
            return None
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0  # 完全上涨
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    @staticmethod
    def calculate_atr(highs: List[float], lows: List[float], 
                      closes: List[float], period: int = 14) -> Optional[float]:
        """
        计算平均真实波幅 (ATR)
        
        Args:
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            period: ATR 周期
            
        Returns:
            最新 ATR 值，数据不足返回 None
        """
        if not all([highs, lows, closes]) or len(highs) < period + 1:
            return None
            
        true_ranges = []
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
            
        atr = sum(true_ranges[-period:]) / period
        
        return round(atr, 4)
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> Optional[float]:
        """
        计算简单移动平均线 (SMA)
        
        Args:
            prices: 价格序列
            period: SMA 周期
            
        Returns:
            最新 SMA 值
        """
        if not prices or len(prices) < period:
            return None
            
        return sum(prices[-period:]) / period
    
    @staticmethod
    def calculate_price_channel(highs: List[float], lows: List[float], 
                                 period: int = 20) -> Optional[Dict[str, float]]:
        """
        计算价格通道（用于第三层滤网入场点确认）
        
        Args:
            highs: 最高价序列
            lows: 最低价序列
            period: 通道周期
            
        Returns:
            包含上轨、中轨、下轨的字典
        """
        if not all([highs, lows]) or len(highs) < period:
            return None
            
        recent_highs = highs[-period:]
        recent_lows = lows[-period:]
        
        upper_channel = max(recent_highs)
        lower_channel = min(recent_lows)
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'upper': round(upper_channel, 2),
            'middle': round(middle_channel, 2),
            'lower': round(lower_channel, 2)
        }


# =============================================================================
# 三重滤网策略引擎
# =============================================================================

class TripleScreenStrategy:
    """三重滤网策略引擎"""
    
    # 趋势方向枚举
    TREND_BULL = "多头趋势"
    TREND_BEAR = "空头趋势"
    TREND_NEUTRAL = "震荡"
    
    # 交易信号枚举
    SIGNAL_LONG = "做多信号"
    SIGNAL_SHORT = "做空信号"
    SIGNAL_CLOSE_LONG = "平多信号"
    SIGNAL_CLOSE_SHORT = "平空信号"
    SIGNAL_WAIT = "观望"
    SIGNAL_NONE = "无信号"
    
    def __init__(self, symbol: str, 
                 ema_short: int = EMA_SHORT,
                 ema_long: int = EMA_LONG,
                 rsi_period: int = RSI_PERIOD,
                 rsi_overbought: int = RSI_OVERBOUGHT,
                 rsi_oversold: int = RSI_OVERSOLD,
                 atr_period: int = ATR_PERIOD):
        """
        初始化三重滤网策略
        
        Args:
            symbol: 交易品种代码
            ema_short: 短期 EMA 周期
            ema_long: 长期 EMA 周期
            rsi_period: RSI 周期
            rsi_overbought: RSI 超买阈值
            rsi_oversold: RSI 超卖阈值
            atr_period: ATR 周期
        """
        self.symbol = symbol
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_period = atr_period
        
        self.indicators = TechnicalIndicators()
        
        # 策略状态
        self.current_trend = self.TREND_NEUTRAL
        self.current_signal = self.SIGNAL_NONE
        self.position_type = None  # None, "long", "short"
        
        # 历史数据
        self.close_prices: List[float] = []
        self.high_prices: List[float] = []
        self.low_prices: List[float] = []
        
        # 计算结果缓存
        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._rsi: Optional[float] = None
        self._atr: Optional[float] = None
        self._price_channel: Optional[Dict[str, float]] = None
    
    def update_data(self, kline_data: Dict[str, Any]) -> None:
        """
        更新 K 线数据
        
        Args:
            kline_data: TQSDK K线数据字典
        """
        # 提取价格数据
        close = kline_data.get('close', [])
        high = kline_data.get('high', [])
        low = kline_data.get('low', [])
        
        if len(close) > 0:
            self.close_prices = list(close)
            self.high_prices = list(high)
            self.low_prices = list(low)
    
    def calculate_filters(self) -> Dict[str, Any]:
        """
        计算三重滤网指标
        
        Returns:
            包含所有指标结果的字典
        """
        if len(self.close_prices) < max(self.ema_long, self.rsi_period + 1, self.atr_period + 1):
            return self._empty_result()
        
        # 第一层滤网：趋势 EMA
        self._ema_fast = self.indicators.calculate_ema(self.close_prices, self.ema_short)
        self._ema_slow = self.indicators.calculate_ema(self.close_prices, self.ema_long)
        
        if self._ema_fast is not None and self._ema_slow is not None:
            if self._ema_fast > self._ema_slow:
                self.current_trend = self.TREND_BULL
            elif self._ema_fast < self._ema_slow:
                self.current_trend = self.TREND_BEAR
            else:
                self.current_trend = self.TREND_NEUTRAL
        else:
            self.current_trend = self.TREND_NEUTRAL
        
        # 第二层滤网：RSI 振荡指标
        self._rsi = self.indicators.calculate_rsi(self.close_prices, self.rsi_period)
        
        # 第三层滤网：ATR 和价格通道
        self._atr = self.indicators.calculate_atr(
            self.high_prices, 
            self.low_prices, 
            self.close_prices, 
            self.atr_period
        )
        
        self._price_channel = self.indicators.calculate_price_channel(
            self.high_prices, 
            self.low_prices
        )
        
        return self._get_indicators_result()
    
    def generate_signal(self, current_price: float) -> Tuple[str, Dict[str, Any]]:
        """
        生成交易信号
        
        Args:
            current_price: 当前价格
            
        Returns:
            (信号类型, 信号详情) 元组
        """
        # 首先计算滤网指标
        indicators = self.calculate_filters()
        
        if not indicators.get('ema_fast') or not indicators.get('rsi'):
            return self.SIGNAL_NONE, {}
        
        signal_details = {
            'trend': self.current_trend,
            'ema_fast': indicators['ema_fast'],
            'ema_slow': indicators['ema_slow'],
            'rsi': indicators['rsi'],
            'atr': indicators['atr'],
            'price_channel': indicators['price_channel'],
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'reason': ""
        }
        
        # 多头趋势 + RSI 超卖回升
        if self.current_trend == self.TREND_BULL:
            if indicators['rsi'] < self.rsi_oversold:
                # RSI 进入超卖区域，可能存在做多机会
                signal_details['reason'] = f"多头趋势中，RSI({indicators['rsi']}) 进入超卖区域"
                
                # 检查价格是否接近支撑位（下轨）
                if indicators['price_channel']:
                    lower = indicators['price_channel']['lower']
                    if current_price <= lower * 1.01:  # 接近下轨1%以内
                        signal_details['entry_price'] = current_price
                        signal_details['stop_loss'] = round(current_price - (indicators['atr'] or 0) * STOP_LOSS_ATR_MULTI, 2)
                        signal_details['take_profit'] = round(current_price + (indicators['atr'] or 0) * TAKE_PROFIT_ATR_MULTI, 2)
                        return self.SIGNAL_LONG, signal_details
                        
            elif indicators['rsi'] > self.rsi_overbought and self.position_type == "long":
                # 多头趋势中 RSI 进入超买，平多
                signal_details['reason'] = f"多头趋势中，RSI({indicators['rsi']}) 进入超买，平多"
                return self.SIGNAL_CLOSE_LONG, signal_details
        
        # 空头趋势 + RSI 超买回落
        elif self.current_trend == self.TREND_BEAR:
            if indicators['rsi'] > self.rsi_overbought:
                # RSI 进入超买区域，可能存在做空机会
                signal_details['reason'] = f"空头趋势中，RSI({indicators['rsi']}) 进入超买区域"
                
                # 检查价格是否接近压力位（上轨）
                if indicators['price_channel']:
                    upper = indicators['price_channel']['upper']
                    if current_price >= upper * 0.99:  # 接近上轨1%以内
                        signal_details['entry_price'] = current_price
                        signal_details['stop_loss'] = round(current_price + (indicators['atr'] or 0) * STOP_LOSS_ATR_MULTI, 2)
                        signal_details['take_profit'] = round(current_price - (indicators['atr'] or 0) * TAKE_PROFIT_ATR_MULTI, 2)
                        return self.SIGNAL_SHORT, signal_details
                        
            elif indicators['rsi'] < self.rsi_oversold and self.position_type == "short":
                # 空头趋势中 RSI 进入超卖，平空
                signal_details['reason'] = f"空头趋势中，RSI({indicators['rsi']}) 进入超卖，平空"
                return self.SIGNAL_CLOSE_SHORT, signal_details
        
        # 震荡市场
        else:
            signal_details['reason'] = "市场趋势不明确，建议观望"
        
        return self.SIGNAL_WAIT, signal_details
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'ema_fast': None,
            'ema_slow': None,
            'rsi': None,
            'atr': None,
            'price_channel': None
        }
    
    def _get_indicators_result(self) -> Dict[str, Any]:
        """获取指标计算结果"""
        return {
            'ema_fast': self._ema_fast,
            'ema_slow': self._ema_slow,
            'rsi': self._rsi,
            'atr': self._atr,
            'price_channel': self._price_channel
        }


# =============================================================================
# 三重滤网交易系统主类
# =============================================================================

class TripleScreenTradingSystem:
    """三重滤网交易系统主类"""
    
    def __init__(self, symbols: List[str] = None, 
                 debug: bool = False):
        """
        初始化交易系统
        
        Args:
            symbols: 交易品种列表
            debug: 是否开启调试模式
        """
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.debug = debug
        
        # TQSDK 连接
        self.api: Optional[TqApi] = None
        self.auth: Optional[TqAuth] = None
        
        # K线数据引用
        self.kline_data: Dict[str, Any] = {}
        self.min_kline_data: Dict[str, Any] = {}
        
        # 策略实例
        self.strategies: Dict[str, TripleScreenStrategy] = {}
        
        # 系统状态
        self.is_running = False
        self.last_update_time = None
        
        # 信号记录
        self.signal_history: List[Dict[str, Any]] = []
        
        # 初始化策略
        for symbol in self.symbols:
            self.strategies[symbol] = TripleScreenStrategy(symbol)
    
    def connect(self, timeout: int = 30) -> bool:
        """
        连接天勤终端
        
        Args:
            timeout: 连接超时时间（秒）
            
        Returns:
            连接是否成功
        """
        try:
            print("=" * 60)
            print("🔗 正在连接天勤终端...")
            print(f"   账号: {TQ_ACCOUNT}")
            print("=" * 60)
            
            self.auth = TqAuth(TQ_ACCOUNT, TQ_PASSWORD)
            self.api = TqApi(auth=self.auth, debug=self.debug)
            
            # 订阅行情数据
            self._subscribe_quotes()
            
            print("✅ 连接成功！")
            return True
            
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False
    
    def _subscribe_quotes(self) -> None:
        """订阅行情数据"""
        print("\n📊 正在订阅行情数据...")
        
        for symbol in self.symbols:
            try:
                # 日线数据（用于趋势判断）
                daily_kline = self.api.get_kline_serial(
                    symbol, 
                    duration_n=DAILY_KLINE[0], 
                    duration_unit=DAILY_KLINE[1]
                )
                
                # 5分钟数据（用于入场点确认）
                min5_kline = self.api.get_kline_serial(
                    symbol,
                    duration_n=MINUTE_5_KLINE[0],
                    duration_unit=MINUTE_5_KLINE[1]
                )
                
                self.kline_data[symbol] = daily_kline
                self.min_kline_data[symbol] = min5_kline
                
                print(f"   ✅ {symbol} - 已订阅")
                
            except Exception as e:
                print(f"   ❌ {symbol} - 订阅失败: {e}")
        
        print()
    
    def run(self, duration: Optional[int] = None) -> None:
        """
        运行交易系统
        
        Args:
            duration: 运行持续时间（秒），None 表示持续运行
        """
        if not self.api:
            print("❌ 请先调用 connect() 方法连接终端")
            return
        
        print("\n" + "=" * 60)
        print("🚀 三重滤网交易系统启动")
        print("=" * 60)
        
        self.is_running = True
        start_time = datetime.now()
        
        try:
            while self.is_running:
                # 等待数据更新
                self.api.wait_update()
                
                # 更新每个品种的策略
                self._update_strategies()
                
                # 显示信号面板
                self._display_signal_board()
                
                # 检查是否超时
                if duration:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed >= duration:
                        self.stop()
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 用户中断，系统正在停止...")
            self.stop()
            
        except Exception as e:
            print(f"\n❌ 系统运行错误: {e}")
            self.stop()
    
    def _update_strategies(self) -> None:
        """更新所有策略的数据和指标"""
        for symbol in self.symbols:
            if symbol not in self.kline_data:
                continue
                
            kline = self.kline_data[symbol]
            
            # 提取 K 线数据
            kline_dict = {
                'close': list(kline['close']),
                'high': list(kline['high']),
                'low': list(kline['low']),
                'open': list(kline['open']),
                'volume': list(kline['volume'])
            }
            
            # 更新策略数据
            strategy = self.strategies[symbol]
            strategy.update_data(kline_dict)
            
            # 获取当前价格
            try:
                quote = self.api.get_quote(symbol)
                current_price = quote.get('last_price', kline_dict['close'][-1] if kline_dict['close'] else 0)
                
                # 生成交易信号
                signal, details = strategy.generate_signal(current_price)
                
                # 记录信号
                if signal not in [TripleScreenStrategy.SIGNAL_NONE, TripleScreenStrategy.SIGNAL_WAIT]:
                    self._record_signal(symbol, signal, current_price, details)
                    
            except Exception as e:
                if self.debug:
                    print(f"更新 {symbol} 时出错: {e}")
    
    def _record_signal(self, symbol: str, signal: str, 
                       price: float, details: Dict[str, Any]) -> None:
        """记录交易信号"""
        record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'price': price,
            'trend': details.get('trend', ''),
            'rsi': details.get('rsi'),
            'atr': details.get('atr'),
            'stop_loss': details.get('stop_loss'),
            'take_profit': details.get('take_profit'),
            'reason': details.get('reason', '')
        }
        
        # 检查是否是重复信号（5分钟内同一品种同一信号）
        is_duplicate = False
        for rec in self.signal_history[-10:]:
            if (rec['symbol'] == symbol and 
                rec['signal'] == signal and
                (datetime.now() - rec['timestamp']).total_seconds() < 300):
                is_duplicate = True
                break
        
        if not is_duplicate:
            self.signal_history.append(record)
            
            # 打印信号通知
            self._print_signal_alert(record)
    
    def _print_signal_alert(self, record: Dict[str, Any]) -> None:
        """打印信号提醒"""
        signal = record['signal']
        
        if signal == TripleScreenStrategy.SIGNAL_LONG:
            prefix = "🟢"
        elif signal == TripleScreenStrategy.SIGNAL_SHORT:
            prefix = "🔴"
        elif signal in [TripleScreenStrategy.SIGNAL_CLOSE_LONG, 
                        TripleScreenStrategy.SIGNAL_CLOSE_SHORT]:
            prefix = "🟡"
        else:
            prefix = "⚪"
        
        print(f"\n{'='*60}")
        print(f"{prefix} 【交易信号】 {record['symbol']}")
        print(f"{'='*60}")
        print(f"   时间: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   信号: {signal}")
        print(f"   价格: {record['price']}")
        print(f"   趋势: {record['trend']}")
        print(f"   RSI:  {record['rsi']}")
        if record['stop_loss']:
            print(f"   止损: {record['stop_loss']}")
        if record['take_profit']:
            print(f"   止盈: {record['take_profit']}")
        print(f"   原因: {record['reason']}")
        print(f"{'='*60}\n")
    
    def _display_signal_board(self) -> None:
        """显示信号看板"""
        # 清屏并显示
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "=" * 80)
        print("📊 三重滤网交易系统 - 实时监控面板")
        print("=" * 80)
        print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # 构建表格数据
        table_data = []
        
        for symbol in self.symbols:
            if symbol not in self.strategies:
                continue
                
            strategy = self.strategies[symbol]
            indicators = strategy.calculate_filters()
            
            # 获取当前价格
            try:
                quote = self.api.get_quote(symbol)
                current_price = quote.get('last_price', '-')
                change_percent = quote.get('change_percent', '-')
                volume = quote.get('volume', '-')  # 成交量
            except:
                current_price = '-'
                change_percent = '-'
                volume = '-'

            # RSI 状态
            rsi = indicators.get('rsi')
            if rsi is not None:
                if rsi > RSI_OVERBOUGHT:
                    rsi_status = f"🔴{rsi:.1f}超买"
                elif rsi < RSI_OVERSOLD:
                    rsi_status = f"🟢{rsi:.1f}超卖"
                else:
                    rsi_status = f"⚪{rsi:.1f}"
            else:
                rsi_status = "-"
            
            # 趋势状态
            trend = strategy.current_trend
            trend_icon = "📈" if trend == TripleScreenStrategy.TREND_BULL else \
                        "📉" if trend == TripleScreenStrategy.TREND_BEAR else "➡️"
            
            # EMA 值
            ema_fast = indicators.get('ema_fast')
            ema_slow = indicators.get('ema_slow')
            
            # ATR 值
            atr = indicators.get('atr')
            
            row = [
                symbol,
                f"{current_price}" if isinstance(current_price, str) else f"{current_price:.2f}",
                f"{change_percent:.2f}%" if isinstance(change_percent, (int, float)) else change_percent,
                trend_icon,
                f"{ema_fast:.2f}" if ema_fast else "-",
                f"{ema_slow:.2f}" if ema_slow else "-",
                rsi_status,
                f"{atr:.4f}" if atr else "-",
                "✅" if strategy.current_signal != TripleScreenStrategy.SIGNAL_NONE else ""
            ]
            
            table_data.append(row)
        
        # 显示表格
        headers = ["品种", "当前价", "涨跌%", "趋势", "EMA12", "EMA26", "RSI", "ATR", "信号"]
        
        if TABULATE_AVAILABLE:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            # 简化输出
            print("\n".join([" | ".join(str(x) for x in row) for row in table_data]))
        
        # 显示图例
        print("\n" + "-" * 80)
        print("【图例】")
        print("  趋势: 📈多头 | 📉空头 | ➡️震荡")
        print("  RSI:  🟢超卖(做多机会) | 🔴超买(做空机会) | ⚪正常")
        print(f"  参数: EMA({EMA_SHORT}/{EMA_LONG}), RSI({RSI_OVERSOLD}/{RSI_OVERBOUGHT}), ATR({ATR_PERIOD})")
        print("=" * 80)
        
        # 显示最新信号
        if self.signal_history:
            print("\n【最近信号】")
            for rec in self.signal_history[-5:]:
                time_str = rec['timestamp'].strftime('%H:%M:%S')
                print(f"  {time_str} | {rec['symbol']:12} | {rec['signal']:10} | {rec['price']}")
        
        self.last_update_time = datetime.now()
    
    def stop(self) -> None:
        """停止交易系统"""
        self.is_running = False
        
        if self.api:
            self.api.close()
            
        print("\n" + "=" * 60)
        print("🛑 三重滤网交易系统已停止")
        print("=" * 60)
        
        # 显示信号统计
        if self.signal_history:
            print(f"\n📈 本次运行共产生 {len(self.signal_history)} 个交易信号")
    
    def get_signal_history(self) -> List[Dict[str, Any]]:
        """获取信号历史记录"""
        return self.signal_history
    
    def get_strategy_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取指定品种的策略状态
        
        Args:
            symbol: 交易品种代码
            
        Returns:
            策略状态字典
        """
        if symbol not in self.strategies:
            return None
            
        strategy = self.strategies[symbol]
        
        return {
            'symbol': symbol,
            'trend': strategy.current_trend,
            'signal': strategy.current_signal,
            'position_type': strategy.position_type,
            'indicators': strategy.calculate_filters()
        }


# =============================================================================
# 演示模式（无需连接）
# =============================================================================

class DemoMode:
    """演示模式 - 使用模拟数据进行策略演示"""
    
    @staticmethod
    def run_demo():
        """运行演示"""
        print("\n" + "=" * 60)
        print("🎯 三重滤网交易系统 - 演示模式")
        print("=" * 60)
        
        # 创建模拟数据
        import random
        random.seed(42)
        
        base_price = 5000.0
        prices = []
        
        for i in range(100):
            change = random.uniform(-50, 50)
            base_price += change
            prices.append(base_price)
        
        # 计算指标
        indicators = TechnicalIndicators()
        
        print("\n【技术指标计算结果】")
        print("-" * 40)
        
        ema_12 = indicators.calculate_ema(prices, 12)
        ema_26 = indicators.calculate_ema(prices, 26)
        rsi = indicators.calculate_rsi(prices, 14)
        
        print(f"  EMA(12): {ema_12:.2f}" if ema_12 else "  EMA(12): 数据不足")
        print(f"  EMA(26): {ema_26:.2f}" if ema_26 else "  EMA(26): 数据不足")
        print(f"  RSI(14): {rsi:.2f}" if rsi else "  RSI(14): 数据不足")
        
        # 模拟信号生成
        print("\n【信号生成演示】")
        print("-" * 40)
        
        # 生成模拟的高低价
        highs = [p + random.uniform(0, 20) for p in prices]
        lows = [p - random.uniform(0, 20) for p in prices]
        
        atr = indicators.calculate_atr(highs, lows, prices, 14)
        channel = indicators.calculate_price_channel(highs, lows, 20)
        
        print(f"  ATR(14): {atr:.4f}" if atr else "  ATR(14): 数据不足")
        
        if channel:
            print(f"  价格通道: 上轨={channel['upper']:.2f}, 中轨={channel['middle']:.2f}, 下轨={channel['lower']:.2f}")
        
        # 三重滤网分析
        print("\n【三重滤网分析】")
        print("-" * 40)
        
        current_price = prices[-1]
        
        # 第一层滤网
        if ema_12 and ema_26:
            if ema_12 > ema_26:
                trend = TripleScreenStrategy.TREND_BULL
            elif ema_12 < ema_26:
                trend = TripleScreenStrategy.TREND_BEAR
            else:
                trend = TripleScreenStrategy.TREND_NEUTRAL
            
            print(f"  第一层(趋势): {trend}")
            print(f"    价格={current_price:.2f} vs EMA12={ema_12:.2f} vs EMA26={ema_26:.2f}")
        
        # 第二层滤网
        if rsi:
            if rsi < RSI_OVERSOLD:
                rsi_status = "超卖(潜在做多信号)"
            elif rsi > RSI_OVERBOUGHT:
                rsi_status = "超买(潜在做空信号)"
            else:
                rsi_status = "正常区间"
            print(f"  第二层(RSI): {rsi:.2f} - {rsi_status}")
        
        # 第三层滤网
        if channel and atr:
            if current_price <= channel['lower'] * 1.01:
                print(f"  第三层(入场点): 价格接近下轨，确认做多入场点")
                print(f"    入场价: {current_price:.2f}")
                print(f"    止损价: {current_price - atr * STOP_LOSS_ATR_MULTI:.2f}")
                print(f"    止盈价: {current_price + atr * TAKE_PROFIT_ATR_MULTI:.2f}")
            elif current_price >= channel['upper'] * 0.99:
                print(f"  第三层(入场点): 价格接近上轨，确认做空入场点")
                print(f"    入场价: {current_price:.2f}")
                print(f"    止损价: {current_price + atr * STOP_LOSS_ATR_MULTI:.2f}")
                print(f"    止盈价: {current_price - atr * TAKE_PROFIT_ATR_MULTI:.2f}")
            else:
                print(f"  第三层(入场点): 价格在通道中部，观望等待")
        
        print("\n" + "=" * 60)
        print("✅ 演示完成")
        print("=" * 60)


# =============================================================================
# 主程序入口
# =============================================================================

def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='期货三重滤网交易系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python triple_screen_trading.py                    # 运行完整系统
  python triple_screen_trading.py --demo             # 运行演示模式
  python triple_screen_trading.py --symbol DCE.j2501 # 指定交易品种
  python triple_screen_trading.py --debug            # 开启调试模式
        """
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='运行演示模式（无需连接天勤终端）'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='指定交易品种代码'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='开启调试模式'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='运行持续时间（秒），默认60秒'
    )
    
    args = parser.parse_args()
    
    # 演示模式
    if args.demo:
        DemoMode.run_demo()
        return
    
    # 正常模式
    symbols = [args.symbol] if args.symbol else None
    
    # 创建交易系统
    trading_system = TripleScreenTradingSystem(
        symbols=symbols,
        debug=args.debug
    )
    
    # 连接天勤终端
    if not trading_system.connect():
        print("\n⚠️ 连接失败，是否使用演示模式?")
        print("   运行: python triple_screen_trading.py --demo")
        sys.exit(1)
    
    # 运行系统
    try:
        trading_system.run(duration=args.duration)
    except KeyboardInterrupt:
        trading_system.stop()


if __name__ == '__main__':
    main()
