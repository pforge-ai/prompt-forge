# ptforge/utils/retry.py
"""
提供灵活的重试策略。
包括指数退避、随机抖动等策略，用于处理临时性错误。

Provides flexible retry strategies.
Includes exponential backoff, jitter, etc. for handling transient errors.
"""

import time
import random
import logging
import functools
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

logger = logging.getLogger(__name__)

# 定义类型变量
# Define type variables
T = TypeVar('T')


class RetryError(Exception):
    """
    重试策略执行失败时抛出的异常。
    
    Exception raised when retry strategy fails.
    """
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        """
        初始化重试错误。
        
        Initialize retry error.
        
        Args:
            message: 错误消息 (Error message)
            last_exception: 最后一次捕获的异常 (Last caught exception)
        """
        super().__init__(message)
        self.last_exception = last_exception


class RetryStrategy:
    """
    重试策略基类。
    定义重试策略的基本接口。
    
    Base retry strategy class.
    Defines the basic interface for retry strategies.
    """
    
    def __init__(self, 
                max_retries: int, 
                retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
                on_retry: Optional[Callable[[int, Exception], None]] = None):
        """
        初始化重试策略。
        
        Initialize retry strategy.
        
        Args:
            max_retries: 最大重试次数 (Maximum number of retries)
            retry_exceptions: 触发重试的异常类型 (Exception types that trigger retries)
            on_retry: 每次重试前调用的回调函数 (Callback function called before each retry)
        """
        self.max_retries = max_retries
        
        # 确保retry_exceptions是列表
        # Ensure retry_exceptions is a list
        if isinstance(retry_exceptions, type) and issubclass(retry_exceptions, Exception):
            self.retry_exceptions = [retry_exceptions]
        else:
            self.retry_exceptions = retry_exceptions
            
        self.on_retry = on_retry
    
    def get_delay(self, attempt: int) -> float:
        """
        获取指定重试次数应等待的延迟时间（秒）。
        
        Get delay time in seconds to wait for the specified retry attempt.
        
        Args:
            attempt: 当前是第几次重试（从1开始） (Current retry attempt number (starting from 1))
            
        Returns:
            等待时间（秒） (Wait time in seconds)
        """
        raise NotImplementedError("Subclasses must implement get_delay()")
    
    def execute(self, 
               func: Callable[..., T], 
               *args: Any, 
               **kwargs: Any) -> T:
        """
        执行函数，如果失败则根据策略重试。
        
        Execute function, retrying according to strategy if it fails.
        
        Args:
            func: 要执行的函数 (Function to execute)
            *args: 传递给函数的位置参数 (Positional arguments to pass to function)
            **kwargs: 传递给函数的关键字参数 (Keyword arguments to pass to function)
            
        Returns:
            函数的返回值 (Return value of function)
            
        Raises:
            RetryError: 如果所有重试都失败 (If all retries fail)
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # 尝试执行函数
                # Try to execute function
                return func(*args, **kwargs)
            except Exception as e:
                # 检查是否是应重试的异常类型
                # Check if exception type should be retried
                if not any(isinstance(e, exc_type) for exc_type in self.retry_exceptions):
                    # 不应重试的异常类型，直接抛出
                    # Exception type that shouldn't be retried, raise directly
                    raise
                    
                last_exception = e
                
                # 如果已经达到最大重试次数，则放弃
                # If maximum retries reached, give up
                if attempt >= self.max_retries:
                    break
                    
                # 计算等待时间
                # Calculate wait time
                delay = self.get_delay(attempt + 1)
                
                # 调用回调（如果有）
                # Call callback if provided
                if self.on_retry:
                    try:
                        self.on_retry(attempt + 1, e)
                    except Exception as callback_error:
                        logger.warning(f"Error in retry callback: {callback_error}")
                
                logger.info(f"Retry {attempt + 1}/{self.max_retries} after error: {e}. Waiting {delay:.2f}s...")
                time.sleep(delay)
        
        # 所有重试都失败
        # All retries failed
        error_message = f"Failed after {self.max_retries} retries"
        if last_exception:
            error_message += f": {last_exception}"
            
        raise RetryError(error_message, last_exception)
    
    def decorate(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        装饰函数，添加重试逻辑。
        
        Decorate function with retry logic.
        
        Args:
            func: 要装饰的函数 (Function to decorate)
            
        Returns:
            装饰后的函数 (Decorated function)
        """
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return self.execute(func, *args, **kwargs)
        
        return wrapper


class ExponentialBackoff(RetryStrategy):
    """
    指数退避重试策略。
    每次重试的等待时间呈指数增长。
    
    Exponential backoff retry strategy.
    Wait time increases exponentially with each retry.
    """
    
    def __init__(self, 
                max_retries: int = 3, 
                initial_delay: float = 1.0,
                max_delay: float = 60.0,
                backoff_factor: float = 2.0,
                retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
                on_retry: Optional[Callable[[int, Exception], None]] = None):
        """
        初始化指数退避重试策略。
        
        Initialize exponential backoff retry strategy.
        
        Args:
            max_retries: 最大重试次数 (Maximum number of retries)
            initial_delay: 初始等待时间（秒） (Initial wait time in seconds)
            max_delay: 最大等待时间（秒） (Maximum wait time in seconds)
            backoff_factor: 退避因子，决定延迟增长的速度 (Backoff factor, determines how quickly delay increases)
            retry_exceptions: 触发重试的异常类型 (Exception types that trigger retries)
            on_retry: 每次重试前调用的回调函数 (Callback function called before each retry)
        """
        super().__init__(max_retries, retry_exceptions, on_retry)
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def get_delay(self, attempt: int) -> float:
        """
        计算指数退避等待时间。
        
        Calculate exponential backoff wait time.
        
        Args:
            attempt: 当前是第几次重试（从1开始） (Current retry attempt number (starting from 1))
            
        Returns:
            等待时间（秒） (Wait time in seconds)
        """
        delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)


class ExponentialBackoffWithJitter(ExponentialBackoff):
    """
    带随机抖动的指数退避重试策略。
    在指数退避的基础上添加随机抖动，避免多个客户端同时重试。
    
    Exponential backoff retry strategy with jitter.
    Adds random jitter on top of exponential backoff to avoid thundering herd problem.
    """
    
    def __init__(self, 
                max_retries: int = 3, 
                initial_delay: float = 1.0,
                max_delay: float = 60.0,
                backoff_factor: float = 2.0,
                jitter_factor: float = 0.1,
                retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
                on_retry: Optional[Callable[[int, Exception], None]] = None):
        """
        初始化带抖动的指数退避重试策略。
        
        Initialize exponential backoff retry strategy with jitter.
        
        Args:
            max_retries: 最大重试次数 (Maximum number of retries)
            initial_delay: 初始等待时间（秒） (Initial wait time in seconds)
            max_delay: 最大等待时间（秒） (Maximum wait time in seconds)
            backoff_factor: 退避因子 (Backoff factor)
            jitter_factor: 抖动因子，控制抖动的最大幅度 (Jitter factor, controls maximum jitter magnitude)
            retry_exceptions: 触发重试的异常类型 (Exception types that trigger retries)
            on_retry: 每次重试前调用的回调函数 (Callback function called before each retry)
        """
        super().__init__(max_retries, initial_delay, max_delay, backoff_factor, retry_exceptions, on_retry)
        self.jitter_factor = jitter_factor
    
    def get_delay(self, attempt: int) -> float:
        """
        计算带抖动的指数退避等待时间。
        
        Calculate exponential backoff wait time with jitter.
        
        Args:
            attempt: 当前是第几次重试（从1开始） (Current retry attempt number (starting from 1))
            
        Returns:
            等待时间（秒） (Wait time in seconds)
        """
        # 先计算基本的指数退避延迟
        # First calculate basic exponential backoff delay
        base_delay = super().get_delay(attempt)
        
        # 添加随机抖动
        # Add random jitter
        jitter_range = base_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        
        # 确保最终延迟不小于0
        # Ensure final delay is not less than 0
        return max(0, base_delay + jitter)


class LinearBackoff(RetryStrategy):
    """
    线性退避重试策略。
    每次重试的等待时间呈线性增长。
    
    Linear backoff retry strategy.
    Wait time increases linearly with each retry.
    """
    
    def __init__(self, 
                max_retries: int = 3, 
                initial_delay: float = 1.0,
                increment: float = 1.0,
                max_delay: float = 60.0,
                retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
                on_retry: Optional[Callable[[int, Exception], None]] = None):
        """
        初始化线性退避重试策略。
        
        Initialize linear backoff retry strategy.
        
        Args:
            max_retries: 最大重试次数 (Maximum number of retries)
            initial_delay: 初始等待时间（秒） (Initial wait time in seconds)
            increment: 每次重试的增量（秒） (Increment for each retry in seconds)
            max_delay: 最大等待时间（秒） (Maximum wait time in seconds)
            retry_exceptions: 触发重试的异常类型 (Exception types that trigger retries)
            on_retry: 每次重试前调用的回调函数 (Callback function called before each retry)
        """
        super().__init__(max_retries, retry_exceptions, on_retry)
        self.initial_delay = initial_delay
        self.increment = increment
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """
        计算线性退避等待时间。
        
        Calculate linear backoff wait time.
        
        Args:
            attempt: 当前是第几次重试（从1开始） (Current retry attempt number (starting from 1))
            
        Returns:
            等待时间（秒） (Wait time in seconds)
        """
        delay = self.initial_delay + (attempt - 1) * self.increment
        return min(delay, self.max_delay)


class FixedDelayWithJitter(RetryStrategy):
    """
    带随机抖动的固定延迟重试策略。
    每次重试使用固定的延迟时间，但添加随机抖动。
    
    Fixed delay retry strategy with jitter.
    Uses a fixed delay time for each retry, but adds random jitter.
    """
    
    def __init__(self, 
                max_retries: int = 3, 
                delay: float = 1.0,
                jitter_factor: float = 0.5,
                retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
                on_retry: Optional[Callable[[int, Exception], None]] = None):
        """
        初始化带抖动的固定延迟重试策略。
        
        Initialize fixed delay retry strategy with jitter.
        
        Args:
            max_retries: 最大重试次数 (Maximum number of retries)
            delay: 基础延迟时间（秒） (Base delay time in seconds)
            jitter_factor: 抖动因子，控制抖动的最大幅度 (Jitter factor, controls maximum jitter magnitude)
            retry_exceptions: 触发重试的异常类型 (Exception types that trigger retries)
            on_retry: 每次重试前调用的回调函数 (Callback function called before each retry)
        """
        super().__init__(max_retries, retry_exceptions, on_retry)
        self.delay = delay
        self.jitter_factor = jitter_factor
    
    def get_delay(self, attempt: int) -> float:
        """
        计算带抖动的固定延迟等待时间。
        
        Calculate fixed delay wait time with jitter.
        
        Args:
            attempt: 当前是第几次重试（从1开始） (Current retry attempt number (starting from 1))
            
        Returns:
            等待时间（秒） (Wait time in seconds)
        """
        # 计算抖动范围
        # Calculate jitter range
        jitter_range = self.delay * self.jitter_factor
        
        # 添加随机抖动
        # Add random jitter
        jitter = random.uniform(-jitter_range, jitter_range)
        
        # 确保最终延迟不小于0
        # Ensure final delay is not less than 0
        return max(0, self.delay + jitter)


def retry(strategy: Optional[RetryStrategy] = None, 
         max_retries: int = 3, 
         initial_delay: float = 1.0,
         backoff_factor: float = 2.0,
         retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception) -> Callable:
    """
    重试装饰器，为函数添加重试逻辑。
    
    Retry decorator to add retry logic to functions.
    
    Args:
        strategy: 重试策略，如果为None则使用指数退避 (Retry strategy, uses exponential backoff if None)
        max_retries: 最大重试次数 (Maximum number of retries)
        initial_delay: 初始等待时间（秒） (Initial wait time in seconds)
        backoff_factor: 退避因子 (Backoff factor)
        retry_exceptions: 触发重试的异常类型 (Exception types that trigger retries)
        
    Returns:
        装饰器函数 (Decorator function)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        nonlocal strategy
        
        # 如果没有提供策略，使用默认的指数退避
        # If no strategy provided, use default exponential backoff
        if strategy is None:
            strategy = ExponentialBackoff(
                max_retries=max_retries,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                retry_exceptions=retry_exceptions
            )
            
        # 使用策略的decorate方法
        # Use strategy's decorate method
        return strategy.decorate(func)
        
    return decorator