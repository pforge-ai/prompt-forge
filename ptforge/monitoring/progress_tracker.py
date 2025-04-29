# ptforge/monitoring/progress_tracker.py
"""
追踪和可视化提示词优化进度。
提供实时监控、统计分析和可视化功能。

Track and visualize prompt optimization progress.
Provides real-time monitoring, statistical analysis, and visualization capabilities.
"""

import os
import time
import json
import logging
import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib and/or NumPy not found. Plotting functionality will be disabled.")
    PLOTTING_AVAILABLE = False


class OptimizationTracker:
    """
    追踪提示词优化进度的核心组件。
    
    Core component for tracking prompt optimization progress.
    """
    
    def __init__(self, 
                experiment_name: str = None,
                save_dir: str = "./optimization_logs",
                save_history: bool = True,
                autosave_interval: int = 5,
                track_memory_usage: bool = False):
        """
        初始化优化追踪器。
        
        Initialize optimization tracker.
        
        Args:
            experiment_name: 实验名称，如果为None则使用时间戳 
                            (Experiment name, uses timestamp if None)
            save_dir: 保存日志和图表的目录 (Directory to save logs and charts)
            save_history: 是否保存历史记录 (Whether to save history)
            autosave_interval: 自动保存间隔（分钟） (Autosave interval in minutes)
            track_memory_usage: 是否追踪内存使用 (Whether to track memory usage)
        """
        self.experiment_name = experiment_name or f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = save_dir
        self.save_history = save_history
        self.autosave_interval = autosave_interval
        self.track_memory_usage = track_memory_usage
        
        # 创建保存目录
        # Create save directory
        self.experiment_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化记录
        # Initialize records
        self.history: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {
            "start_time": time.time(),
            "last_update_time": time.time(),
            "last_save_time": time.time(),
            "steps": 0,
            "accepted_steps": 0,
            "start_score": None,
            "current_score": None,
            "best_score": None,
            "best_step": None,
            "best_improvement": 0,
            "actions_by_type": {},
            "actions_by_section": {},
            "consecutive_no_improvement": 0,
        }
        
        logger.info(f"Initialized OptimizationTracker for experiment '{self.experiment_name}'")
        
        # 如果启用了追踪内存使用
        # If memory usage tracking is enabled
        if self.track_memory_usage:
            try:
                import psutil
                self.psutil_available = True
                self.process = psutil.Process(os.getpid())
                logger.info("Memory usage tracking enabled")
            except ImportError:
                logger.warning("psutil not found. Memory usage tracking will be disabled.")
                self.psutil_available = False
    
    def update(self, step_info: Dict[str, Any]) -> None:
        """
        更新优化进度。
        
        Update optimization progress.
        
        Args:
            step_info: 步骤信息 (Step information)
        """
        # 添加到历史记录
        # Add to history
        self.history.append(step_info)
        
        # 更新统计信息
        # Update statistics
        self.stats["steps"] += 1
        self.stats["last_update_time"] = time.time()
        
        # 记录第一步的初始分数
        # Record initial score on first step
        if self.stats["start_score"] is None and "before_score" in step_info:
            self.stats["start_score"] = step_info["before_score"]
            self.stats["current_score"] = step_info["before_score"]
            self.stats["best_score"] = step_info["before_score"]
        
        # 更新当前和最佳分数
        # Update current and best scores
        if "after_score" in step_info:
            if step_info.get("accepted", True):
                self.stats["current_score"] = step_info["after_score"]
                self.stats["accepted_steps"] += 1
            
            if (self.stats["best_score"] is None or 
                step_info["after_score"] > self.stats["best_score"]):
                self.stats["best_score"] = step_info["after_score"]
                self.stats["best_step"] = self.stats["steps"]
                
                # 计算相对改进
                # Calculate relative improvement
                if self.stats["start_score"] and self.stats["start_score"] != 0:
                    self.stats["best_improvement"] = (
                        (self.stats["best_score"] - self.stats["start_score"]) / 
                        abs(self.stats["start_score"]) * 100
                    )
                else:
                    self.stats["best_improvement"] = 0
        
        # 更新连续无改进计数
        # Update consecutive no improvement count
        if step_info.get("accepted", True) and "before_score" in step_info and "after_score" in step_info:
            if step_info["after_score"] > step_info["before_score"]:
                self.stats["consecutive_no_improvement"] = 0
            else:
                self.stats["consecutive_no_improvement"] += 1
        
        # 更新动作统计
        # Update action statistics
        if "action" in step_info:
            action = step_info["action"]
            
            # 按动作类型统计
            # Statistics by action type
            action_type = action.get("action_type", "UNKNOWN")
            self.stats["actions_by_type"][action_type] = self.stats["actions_by_type"].get(action_type, 0) + 1
            
            # 按目标部分统计
            # Statistics by target section
            section = action.get("target_section", "UNKNOWN")
            self.stats["actions_by_section"][section] = self.stats["actions_by_section"].get(section, 0) + 1
        
        # 追踪内存使用
        # Track memory usage
        if self.track_memory_usage and self.psutil_available:
            memory_info = self.process.memory_info()
            step_info["memory_rss_mb"] = memory_info.rss / (1024 * 1024)
            step_info["memory_vms_mb"] = memory_info.vms / (1024 * 1024)
        
        # 检查是否需要自动保存
        # Check if autosave is needed
        if (self.save_history and 
            time.time() - self.stats["last_save_time"] > self.autosave_interval * 60):
            self.save()
            self.stats["last_save_time"] = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取优化进度概要。
        
        Get optimization progress summary.
        
        Returns:
            进度概要字典 (Progress summary dictionary)
        """
        elapsed_time = time.time() - self.stats["start_time"]
        
        return {
            "experiment_name": self.experiment_name,
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_formatted": self._format_time(elapsed_time),
            "steps": self.stats["steps"],
            "accepted_steps": self.stats["accepted_steps"],
            "acceptance_rate": self.stats["accepted_steps"] / max(1, self.stats["steps"]),
            "start_score": self.stats["start_score"],
            "current_score": self.stats["current_score"],
            "best_score": self.stats["best_score"],
            "best_step": self.stats["best_step"],
            "relative_improvement": self.stats["best_improvement"],
            "steps_since_last_improvement": self.stats["consecutive_no_improvement"],
            "most_common_action_type": self._get_most_common(self.stats["actions_by_type"]),
            "most_targeted_section": self._get_most_common(self.stats["actions_by_section"]),
            "steps_per_minute": self.stats["steps"] / max(1, elapsed_time / 60),
        }
    
    def _get_most_common(self, counter: Dict[str, int]) -> Tuple[str, int]:
        """
        获取计数器中最常见的项。
        
        Get most common item in counter.
        
        Args:
            counter: 计数器字典 (Counter dictionary)
            
        Returns:
            (最常见项, 计数) ((most common item, count))
        """
        if not counter:
            return ("N/A", 0)
            
        return max(counter.items(), key=lambda x: x[1])
    
    def _format_time(self, seconds: float) -> str:
        """
        格式化时间（秒）为可读字符串。
        
        Format time in seconds to readable string.
        
        Args:
            seconds: 秒数 (Seconds)
            
        Returns:
            格式化的时间字符串 (Formatted time string)
        """
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    
    def save(self) -> str:
        """
        保存跟踪数据到文件。
        
        Save tracking data to file.
        
        Returns:
            保存的文件路径 (Path to saved file)
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存摘要
        # Save summary
        summary_path = os.path.join(self.experiment_dir, f"summary_{timestamp}.json")
        with open(summary_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)
        
        # 保存历史
        # Save history
        if self.save_history:
            history_path = os.path.join(self.experiment_dir, f"history_{timestamp}.json")
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        
        logger.info(f"Saved optimization tracking data to {self.experiment_dir}")
        return summary_path
    
    def plot_progress(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        绘制优化进度图表。
        
        Plot optimization progress chart.
        
        Args:
            save_path: 保存图表的路径，如果为None则使用默认路径
                      (Path to save chart, uses default path if None)
            
        Returns:
            Matplotlib Figure对象，如果plotting不可用则返回None 
            (Matplotlib Figure object, or None if plotting is not available)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting is not available due to missing dependencies.")
            return None
            
        if not self.history:
            logger.warning("No history to plot.")
            return None
            
        # 提取数据
        # Extract data
        steps = list(range(1, len(self.history) + 1))
        before_scores = [step.get("before_score", None) for step in self.history]
        after_scores = [step.get("after_score", None) for step in self.history]
        rewards = [step.get("reward", 0) for step in self.history]
        
        # 创建图表
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 绘制分数
        # Plot scores
        ax1.plot(steps, before_scores, 'b--', alpha=0.7, label='Before Score')
        ax1.plot(steps, after_scores, 'g-', label='After Score')
        
        # 添加最佳分数标记
        # Add best score marker
        if self.stats["best_step"] is not None:
            best_idx = self.stats["best_step"] - 1
            if 0 <= best_idx < len(self.history):
                best_score = self.stats["best_score"]
                ax1.plot(self.stats["best_step"], best_score, 'ro', markersize=8, label=f'Best Score: {best_score:.4f}')
        
        ax1.set_title(f'Optimization Progress - {self.experiment_name}')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制奖励
        # Plot rewards
        ax2.plot(steps, rewards, 'r-', label='Reward')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax2.set_xlabel('Optimization Steps')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        # Save figure
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.experiment_dir, f"progress_{timestamp}.png")
            
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved progress plot to {save_path}")
            
        return fig
    
    def plot_action_statistics(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        绘制动作统计图表。
        
        Plot action statistics chart.
        
        Args:
            save_path: 保存图表的路径，如果为None则使用默认路径
                      (Path to save chart, uses default path if None)
            
        Returns:
            Matplotlib Figure对象，如果plotting不可用则返回None
            (Matplotlib Figure object, or None if plotting is not available)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting is not available due to missing dependencies.")
            return None
            
        if not self.history:
            logger.warning("No history to plot.")
            return None
            
        # 创建图表
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制动作类型分布
        # Plot action type distribution
        action_types = self.stats["actions_by_type"]
        if action_types:
            labels = list(action_types.keys())
            values = list(action_types.values())
            
            ax1.bar(labels, values, color='skyblue')
            ax1.set_title('Actions by Type')
            ax1.set_ylabel('Count')
            if len(labels) > 5:
                ax1.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center', va='center')
            
        # 绘制目标部分分布
        # Plot target section distribution
        sections = self.stats["actions_by_section"]
        if sections:
            labels = list(sections.keys())
            values = list(sections.values())
            
            ax2.bar(labels, values, color='lightgreen')
            ax2.set_title('Actions by Target Section')
            ax2.set_ylabel('Count')
            if len(labels) > 5:
                ax2.set_xticklabels(labels, rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center')
            
        plt.tight_layout()
        
        # 保存图表
        # Save figure
        if save_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.experiment_dir, f"action_stats_{timestamp}.png")
            
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved action statistics plot to {save_path}")
            
        return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        生成优化报告。
        
        Generate optimization report.
        
        Args:
            save_path: 保存报告的路径，如果为None则使用默认路径
                      (Path to save report, uses default path if None)
            
        Returns:
            报告文件路径，如果生成失败则返回None
            (Report file path, or None if generation fails)
        """
        try:
            if save_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.experiment_dir, f"report_{timestamp}.md")
                
            summary = self.get_summary()
            
            # 绘制图表并保存
            # Draw plots and save
            plot_path = os.path.join(self.experiment_dir, f"progress_{timestamp}.png")
            self.plot_progress(plot_path)
            
            stats_path = os.path.join(self.experiment_dir, f"action_stats_{timestamp}.png")
            self.plot_action_statistics(stats_path)
            
            # 生成Markdown报告
            # Generate Markdown report
            with open(save_path, "w") as f:
                f.write(f"# Optimization Report: {self.experiment_name}\n\n")
                f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Summary\n\n")
                f.write(f"- **Duration:** {summary['elapsed_time_formatted']}\n")
                f.write(f"- **Total Steps:** {summary['steps']}\n")
                f.write(f"- **Accepted Steps:** {summary['accepted_steps']} ({summary['acceptance_rate']:.1%})\n")
                f.write(f"- **Start Score:** {summary['start_score']:.4f}\n")
                f.write(f"- **Best Score:** {summary['best_score']:.4f}\n")
                f.write(f"- **Improvement:** {summary['relative_improvement']:.2f}%\n")
                f.write(f"- **Steps Per Minute:** {summary['steps_per_minute']:.2f}\n\n")
                
                f.write("## Performance\n\n")
                f.write(f"![Progress Chart]({os.path.basename(plot_path)})\n\n")
                
                f.write("## Action Statistics\n\n")
                f.write(f"- **Most Common Action Type:** {summary['most_common_action_type'][0]} (used {summary['most_common_action_type'][1]} times)\n")
                f.write(f"- **Most Targeted Section:** {summary['most_targeted_section'][0]} (targeted {summary['most_targeted_section'][1]} times)\n\n")
                f.write(f"![Action Statistics]({os.path.basename(stats_path)})\n\n")
                
                if self.history:
                    f.write("## Best Performing Actions\n\n")
                    # 找出最有效的动作
                    # Find most effective actions
                    effective_actions = sorted(
                        [(i, h) for i, h in enumerate(self.history) if "before_score" in h and "after_score" in h], 
                        key=lambda x: x[1]["after_score"] - x[1]["before_score"],
                        reverse=True
                    )[:5]
                    
                    if effective_actions:
                        f.write("| Step | Action Type | Target Section | Improvement |\n")
                        f.write("|------|------------|----------------|-------------|\n")
                        for i, action in effective_actions:
                            improvement = action["after_score"] - action["before_score"]
                            action_type = action["action"].get("action_type", "UNKNOWN")
                            section = action["action"].get("target_section", "UNKNOWN")
                            f.write(f"| {i+1} | {action_type} | {section} | {improvement:.4f} |\n")
                        f.write("\n")
                
            logger.info(f"Generated optimization report at {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            return None


class LiveProgressMonitor:
    """
    实时监控优化进度的组件。
    支持在优化期间更新进度条、指标显示等。
    
    Component for monitoring optimization progress in real-time.
    Supports updating progress bars, metric displays, etc. during optimization.
    """
    
    def __init__(self, tracker: OptimizationTracker, update_interval: float = 1.0):
        """
        初始化实时进度监控器。
        
        Initialize live progress monitor.
        
        Args:
            tracker: 优化追踪器 (Optimization tracker)
            update_interval: 更新间隔（秒） (Update interval in seconds)
        """
        self.tracker = tracker
        self.update_interval = update_interval
        self.last_update_time = 0
        self.enable_tqdm = self._check_tqdm_available()
        self.progress_bar = None
        
    def _check_tqdm_available(self) -> bool:
        """
        检查tqdm是否可用。
        
        Check if tqdm is available.
        
        Returns:
            tqdm是否可用 (Whether tqdm is available)
        """
        try:
            import tqdm
            return True
        except ImportError:
            return False
    
    def start(self, total_steps: int) -> None:
        """
        启动监控。
        
        Start monitoring.
        
        Args:
            total_steps: 总步数 (Total steps)
        """
        if self.enable_tqdm:
            from tqdm import tqdm
            self.progress_bar = tqdm(total=total_steps, desc="Optimizing")
        
        self.last_update_time = time.time()
    
    def update(self, force: bool = False) -> None:
        """
        更新显示。
        
        Update display.
        
        Args:
            force: 是否强制更新，不考虑时间间隔 (Whether to force update regardless of time interval)
        """
        current_time = time.time()
        
        # 检查是否需要更新
        # Check if update is needed
        if not force and current_time - self.last_update_time < self.update_interval:
            return
            
        # 更新进度条
        # Update progress bar
        if self.progress_bar:
            summary = self.tracker.get_summary()
            self.progress_bar.n = summary["steps"]
            
            # 更新描述
            # Update description
            desc = f"Score: {summary['current_score']:.4f}"
            if summary["best_score"] is not None:
                desc += f" | Best: {summary['best_score']:.4f} ({summary['relative_improvement']:.1f}%)"
                
            self.progress_bar.set_description(desc)
            self.progress_bar.refresh()
            
        self.last_update_time = current_time
    
    def stop(self) -> None:
        """
        停止监控。
        
        Stop monitoring.
        """
        if self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None