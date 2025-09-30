"""
Automated Color Contrast Monitoring System
Runs periodic checks and generates alerts for accessibility issues
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
from sandbox.utils.color_contrast_checker import ColorContrastChecker


class ContrastMonitor:
    """Automated system to monitor color contrast in the dashboard"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging for the contrast monitor"""
        log_file = self.log_dir / "contrast_monitor.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger("ContrastMonitor")

    def run_contrast_check(self) -> None:
        """Run a contrast check and log results"""
        try:
            self.logger.info("Starting automated contrast check...")

            checker = ColorContrastChecker()
            results = checker.check_dashboard_combinations()

            # Count results by category
            excellent = len([r for r in results if r.passes_aaa])
            good = len([r for r in results if r.passes_aa and not r.passes_aaa])
            poor = len([r for r in results if not r.passes_aa])

            # Log summary
            self.logger.info(
                f"Contrast check completed - Total: {len(results)}, "
                f"Excellent: {excellent}, Good: {good}, Poor: {poor}"
            )

            # Alert on issues
            if poor > 0:
                self.logger.warning(
                    f"⚠️  Found {poor} color combinations with poor contrast!"
                )

                # Log specific issues
                for result in results:
                    if not result.passes_aa:
                        self.logger.warning(
                            f"   {result.background} on {result.foreground} "
                            f"- Ratio: {result.ratio:.2f}"
                        )

                # Save detailed report
                report_file = (
                    self.log_dir
                    / f"contrast_issues_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                with open(report_file, "w") as f:
                    f.write(checker.generate_report())

                self.logger.info(f"Detailed report saved to: {report_file}")

            else:
                self.logger.info(
                    "✅ All color combinations meet accessibility standards!"
                )

        except Exception as e:
            self.logger.error(f"Error during contrast check: {e}")

    def start_monitoring(self) -> None:
        """Start the automated monitoring system"""
        self.logger.info("Starting contrast monitoring system...")

        # Schedule checks
        schedule.every().day.at("09:00").do(
            self.run_contrast_check
        )  # Daily morning check
        schedule.every().monday.at("14:00").do(
            self.run_contrast_check
        )  # Weekly detailed check

        # Run initial check
        self.run_contrast_check()

        # Keep the monitor running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute for scheduled jobs

    def run_single_check(self) -> dict:
        """Run a single contrast check and return results"""
        try:
            checker = ColorContrastChecker()
            results = checker.check_dashboard_combinations()

            return {
                "timestamp": datetime.now().isoformat(),
                "total": len(results),
                "excellent": len([r for r in results if r.passes_aaa]),
                "good": len([r for r in results if r.passes_aa and not r.passes_aaa]),
                "poor": len([r for r in results if not r.passes_aa]),
                "issues": [
                    {
                        "background": r.background,
                        "foreground": r.foreground,
                        "ratio": r.ratio,
                        "recommendation": r.recommendation,
                    }
                    for r in results
                    if not r.passes_aa
                ],
            }

        except Exception as e:
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}


def main() -> None:
    """Run the contrast monitor"""
    monitor = ContrastMonitor()

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Run single check
        results = monitor.run_single_check()
        print(f"Contrast Check Results: {results}")
    else:
        # Start continuous monitoring
        monitor.start_monitoring()


if __name__ == "__main__":
    main()
