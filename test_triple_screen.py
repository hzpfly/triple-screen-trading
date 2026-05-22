#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货三重滤网交易系统 - 单元测试
测试范围：TechnicalIndicators、TripleScreenStrategy 核心逻辑
不依赖 TQSDK 真实连接，全部使用模拟数据
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triple_screen_trading import (
    TechnicalIndicators,
    TripleScreenStrategy,
    EMA_SHORT, EMA_LONG,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    ATR_PERIOD,
    STOP_LOSS_ATR_MULTI, TAKE_PROFIT_ATR_MULTI,
    DEFAULT_SYMBOLS,
)


# =============================================================================
# 辅助函数
# =============================================================================

def make_kline_data(
    close_prices: List[float],
    high_offset: float = 10.0,
    low_offset: float = 10.0
) -> Dict[str, Any]:
    highs = [c + high_offset for c in close_prices]
    lows = [c - low_offset for c in close_prices]
    return {
        'close': close_prices,
        'high': highs,
        'low': lows,
        'open': close_prices[:],
        'volume': [1000] * len(close_prices),
    }


def generate_trend_prices(
    start: float,
    length: int,
    trend: str = "up",
    volatility: float = 5.0
) -> List[float]:
    import random
    random.seed(42)
    prices = [start]
    for i in range(1, length):
        if trend == "up":
            bias = 2.0
        elif trend == "down":
            bias = -2.0
        else:
            bias = 0.0
        change = bias + random.uniform(-volatility, volatility)
        prices.append(prices[-1] + change)
    return prices


# =============================================================================
# TechnicalIndicators 测试
# =============================================================================

class TestEMA(unittest.TestCase):
    def test_insufficient_data(self):
        self.assertIsNone(TechnicalIndicators.calculate_ema([100.0, 101.0, 102.0], period=12))

    def test_sufficient_data(self):
        prices = generate_trend_prices(100.0, 30, trend="up")
        result = TechnicalIndicators.calculate_ema(prices, period=12)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)

    def test_trending_up(self):
        prices = [100.0 + i for i in range(30)]
        ema = TechnicalIndicators.calculate_ema(prices, period=12)
        self.assertIsNotNone(ema)
        self.assertGreater(ema, 110.0)


class TestRSI(unittest.TestCase):
    def test_insufficient_data(self):
        prices = [100.0] * 10
        self.assertIsNone(TechnicalIndicators.calculate_rsi(prices, period=14))

    def test_all_up(self):
        prices = [100.0 + i * 1.0 for i in range(20)]
        result = TechnicalIndicators.calculate_rsi(prices, period=14)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 100.0, delta=1.0)

    def test_all_down(self):
        prices = [100.0 - i * 1.0 for i in range(20)]
        result = TechnicalIndicators.calculate_rsi(prices, period=14)
        self.assertIsNotNone(result)
        self.assertLess(result, 5.0)

    def test_range(self):
        prices = generate_trend_prices(100.0, 30, volatility=3.0)
        result = TechnicalIndicators.calculate_rsi(prices, period=14)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 100.0)


class TestATR(unittest.TestCase):
    def test_insufficient_data(self):
        highs = [110.0] * 10
        lows = [90.0] * 10
        closes = [100.0] * 10
        self.assertIsNone(TechnicalIndicators.calculate_atr(highs, lows, closes, period=14))

    def test_constant_range(self):
        closes = [100.0 + i for i in range(20)]
        highs = [c + 10.0 for c in closes]
        lows = [c - 10.0 for c in closes]
        result = TechnicalIndicators.calculate_atr(highs, lows, closes, period=14)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 20.0, delta=1.0)

    def test_positive(self):
        closes = generate_trend_prices(100.0, 30)
        highs = [c + 5.0 for c in closes]
        lows = [c - 5.0 for c in closes]
        result = TechnicalIndicators.calculate_atr(highs, lows, closes, period=14)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.0)


class TestPriceChannel(unittest.TestCase):
    def test_insufficient_data(self):
        self.assertIsNone(TechnicalIndicators.calculate_price_channel([110.0, 112.0], [90.0, 88.0], period=20))

    def test_values(self):
        highs = [100.0 + i for i in range(25)]
        lows = [90.0 + i for i in range(25)]
        result = TechnicalIndicators.calculate_price_channel(highs, lows, period=20)
        self.assertIsNotNone(result)
        self.assertGreater(result['upper'], result['middle'])
        self.assertGreater(result['middle'], result['lower'])

    def test_upper_is_max(self):
        highs = list(range(100, 120))
        lows = list(range(80, 100))
        result = TechnicalIndicators.calculate_price_channel(highs, lows, period=10)
        self.assertEqual(result['upper'], max(highs[-10:]))


class TestSMA(unittest.TestCase):
    def test_insufficient_data(self):
        self.assertIsNone(TechnicalIndicators.calculate_sma([100.0, 101.0], period=5))

    def test_correct(self):
        prices = [100.0, 102.0, 104.0, 106.0, 108.0]
        self.assertEqual(TechnicalIndicators.calculate_sma(prices, period=5), 104.0)


# =============================================================================
# TripleScreenStrategy 测试
# =============================================================================

class TestStrategyInit(unittest.TestCase):
    def test_default(self):
        s = TripleScreenStrategy("DCE.m2505")
        self.assertEqual(s.symbol, "DCE.m2505")
        self.assertEqual(s.ema_short, EMA_SHORT)
        self.assertEqual(s.current_trend, TripleScreenStrategy.TREND_NEUTRAL)

    def test_custom_params(self):
        s = TripleScreenStrategy("DCE.j2505", ema_short=5, ema_long=20, rsi_oversold=20, rsi_overbought=80)
        self.assertEqual(s.ema_short, 5)
        self.assertEqual(s.rsi_oversold, 20)


class TestStrategyUpdateData(unittest.TestCase):
    def test_update(self):
        s = TripleScreenStrategy("DCE.m2505")
        kline = make_kline_data([100.0, 101.0, 102.0])
        s.update_data(kline)
        self.assertEqual(len(s.close_prices), 3)


class TestStrategyCalculateFilters(unittest.TestCase):
    def test_insufficient_data(self):
        s = TripleScreenStrategy("DCE.m2505")
        kline = make_kline_data([100.0] * 10)
        s.update_data(kline)
        result = s.calculate_filters()
        self.assertIsNone(result['ema_fast'])

    def test_sufficient_data_bull(self):
        s = TripleScreenStrategy("DCE.m2505")
        prices = generate_trend_prices(100.0, 50, trend="up")
        kline = make_kline_data(prices)
        s.update_data(kline)
        result = s.calculate_filters()
        self.assertIsNotNone(result['ema_fast'])
        self.assertEqual(s.current_trend, TripleScreenStrategy.TREND_BULL)


class TestStrategyGenerateSignal(unittest.TestCase):
    """
    测试 generate_signal 的信号逻辑。
    用 mock 替代 calculate_filters，避免构造复杂价格序列。
    """

    @patch.object(TripleScreenStrategy, 'calculate_filters')
    def test_signal_none_when_insufficient(self, mock_calc):
        s = TripleScreenStrategy("DCE.m2505")
        s.close_prices = [100.0] * 10  # 数据不足
        mock_calc.return_value = {'ema_fast': None, 'rsi': None}
        signal, details = s.generate_signal(100.0)
        self.assertEqual(signal, TripleScreenStrategy.SIGNAL_NONE)

    @patch.object(TripleScreenStrategy, 'calculate_filters')
    def test_signal_long(self, mock_calc):
        """多头趋势 + RSI 超卖 + 价格接近下轨 → 做多信号"""
        s = TripleScreenStrategy("DCE.m2505", rsi_oversold=30, rsi_overbought=70)
        s.close_prices = [100.0] * 30
        s.high_prices = [110.0] * 30
        s.low_prices = [90.0] * 30
        s.current_trend = TripleScreenStrategy.TREND_BULL
        s._atr = 2.0
        s._price_channel = {'upper': 120.0, 'middle': 110.0, 'lower': 100.0}

        mock_calc.return_value = {
            'ema_fast': 105.0,
            'ema_slow': 100.0,
            'rsi': 25.0,
            'atr': 2.0,
            'price_channel': s._price_channel,
        }

        signal, details = s.generate_signal(100.5)  # 接近下轨
        self.assertEqual(signal, TripleScreenStrategy.SIGNAL_LONG)
        self.assertIsNotNone(details.get('entry_price'))
        self.assertIsNotNone(details.get('stop_loss'))
        self.assertIsNotNone(details.get('take_profit'))

    @patch.object(TripleScreenStrategy, 'calculate_filters')
    def test_signal_short(self, mock_calc):
        """空头趋势 + RSI 超买 + 价格接近上轨 → 做空信号"""
        s = TripleScreenStrategy("DCE.m2505", rsi_oversold=30, rsi_overbought=70)
        s.close_prices = [200.0] * 30
        s.high_prices = [210.0] * 30
        s.low_prices = [190.0] * 30
        s.current_trend = TripleScreenStrategy.TREND_BEAR
        s._atr = 2.0
        s._price_channel = {'upper': 200.0, 'middle': 190.0, 'lower': 180.0}

        mock_calc.return_value = {
            'ema_fast': 195.0,
            'ema_slow': 200.0,
            'rsi': 75.0,
            'atr': 2.0,
            'price_channel': s._price_channel,
        }

        signal, details = s.generate_signal(199.5)  # 接近上轨
        self.assertEqual(signal, TripleScreenStrategy.SIGNAL_SHORT)
        self.assertIsNotNone(details.get('entry_price'))

    @patch.object(TripleScreenStrategy, 'calculate_filters')
    def test_signal_close_long_when_overbought(self, mock_calc):
        """多头趋势中 RSI 超买 → 平多信号"""
        s = TripleScreenStrategy("DCE.m2505", rsi_overbought=70)
        s.position_type = "long"
        s.current_trend = TripleScreenStrategy.TREND_BULL

        mock_calc.return_value = {
            'ema_fast': 105.0,
            'ema_slow': 100.0,
            'rsi': 75.0,
            'atr': 2.0,
            'price_channel': None,
        }

        signal, details = s.generate_signal(105.0)
        self.assertEqual(signal, TripleScreenStrategy.SIGNAL_CLOSE_LONG)

    @patch.object(TripleScreenStrategy, 'calculate_filters')
    def test_signal_close_short_when_oversold(self, mock_calc):
        """空头趋势中 RSI 超卖 → 平空信号"""
        s = TripleScreenStrategy("DCE.m2505", rsi_oversold=30)
        s.position_type = "short"
        s.current_trend = TripleScreenStrategy.TREND_BEAR

        mock_calc.return_value = {
            'ema_fast': 95.0,
            'ema_slow': 100.0,
            'rsi': 25.0,
            'atr': 2.0,
            'price_channel': None,
        }

        signal, details = s.generate_signal(95.0)
        self.assertEqual(signal, TripleScreenStrategy.SIGNAL_CLOSE_SHORT)

    @patch.object(TripleScreenStrategy, 'calculate_filters')
    def test_signal_wait_when_neutral(self, mock_calc):
        """震荡趋势 → 观望信号"""
        s = TripleScreenStrategy("DCE.m2505")
        s.current_trend = TripleScreenStrategy.TREND_NEUTRAL

        mock_calc.return_value = {
            'ema_fast': 100.0,
            'ema_slow': 100.0,
            'rsi': 50.0,
            'atr': 2.0,
            'price_channel': None,
        }

        signal, _ = s.generate_signal(100.0)
        # 震荡时 RSI 不超买也不超卖 → 返回 WAIT
        self.assertIn(signal, (
            TripleScreenStrategy.SIGNAL_WAIT,
            TripleScreenStrategy.SIGNAL_NONE,
        ))


class TestStrategyEdgeCases(unittest.TestCase):
    def test_empty_prices(self):
        s = TripleScreenStrategy("DCE.m2505")
        s.update_data({'close': [], 'high': [], 'low': [], 'open': [], 'volume': []})
        result = s.calculate_filters()
        self.assertIsNone(result['ema_fast'])

    def test_position_type_tracking(self):
        s = TripleScreenStrategy("DCE.m2505")
        self.assertIsNone(s.position_type)
        s.position_type = "long"
        self.assertEqual(s.position_type, "long")


# =============================================================================
# 集成测试
# =============================================================================

class TestIntegration(unittest.TestCase):
    def test_full_flow_bull(self):
        s = TripleScreenStrategy("DCE.m2505")
        prices = generate_trend_prices(100.0, 60, trend="up")
        kline = make_kline_data(prices)
        s.update_data(kline)
        indicators = s.calculate_filters()
        self.assertIsNotNone(indicators['ema_fast'])
        signal, details = s.generate_signal(prices[-1])
        self.assertIn('trend', details)

    def test_indicators_consistency(self):
        s = TripleScreenStrategy("DCE.m2505")
        prices = generate_trend_prices(100.0, 50, trend="up")
        kline = make_kline_data(prices)
        s.update_data(kline)
        r1 = s.calculate_filters()
        r2 = s.calculate_filters()
        self.assertEqual(r1['ema_fast'], r2['ema_fast'])


# =============================================================================
# 配置测试
# =============================================================================

class TestConfig(unittest.TestCase):
    def test_default_symbols_not_empty(self):
        self.assertTrue(len(DEFAULT_SYMBOLS) > 0)

    def test_ema_params_valid(self):
        self.assertGreater(EMA_SHORT, 0)
        self.assertGreater(EMA_LONG, EMA_SHORT)

    def test_rsi_params_valid(self):
        self.assertGreater(RSI_OVERBOUGHT, RSI_OVERSOLD)

    def test_atr_multipliers_positive(self):
        self.assertGreater(STOP_LOSS_ATR_MULTI, 0)
        self.assertGreater(TAKE_PROFIT_ATR_MULTI, 0)


# =============================================================================
# 主入口
# =============================================================================

def run_tests(verbosity=2):
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestEMA,
        TestRSI,
        TestATR,
        TestPriceChannel,
        TestSMA,
        TestStrategyInit,
        TestStrategyUpdateData,
        TestStrategyCalculateFilters,
        TestStrategyGenerateSignal,
        TestStrategyEdgeCases,
        TestIntegration,
        TestConfig,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='三重滤网交易系统单元测试')
    parser.add_argument('--verbosity', '-v', type=int, default=2)
    args = parser.parse_args()
    success = run_tests(verbosity=args.verbosity)
    sys.exit(0 if success else 1)
