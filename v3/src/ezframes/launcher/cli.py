from __future__ import annotations

import argparse
import logging
import time

from ezframes import __version__ as EZFRAMES_VERSION
from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.common.logging_utils import configure_logging
from ezframes.common.state import load_install_state, save_install_state
from ezframes.launcher.auth_service import AuthService
from ezframes.launcher.migration_service import MigrationService
from ezframes.launcher.process_service import ProcessService
from ezframes.launcher.shortcut_service import ShortcutService
from ezframes.launcher.update_service import UpdateService


log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EzFrames v3 launcher")
    parser.add_argument("--check-only", action="store_true", help="Check migration/auth/update and exit.")
    parser.add_argument("--log-level", default="INFO", help="Log level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths = AppPaths.default()
    paths.ensure_layout()
    configure_logging(paths.logs_dir, level=args.log_level)

    runtime_cfg = RuntimeConfig.from_env()
    state = load_install_state(paths.state_dir)
    shortcuts = ShortcutService(paths)
    shortcuts.repair_shortcuts()

    migration = MigrationService(paths)
    state = migration.run_if_needed(state)
    if not state.installed_version:
        state.installed_version = EZFRAMES_VERSION
        log.info("Initialized install state version from bundled app: %s", EZFRAMES_VERSION)
    save_install_state(paths.state_dir, state)
    shortcuts.repair_shortcuts()

    update_service = UpdateService(paths, runtime_cfg)
    update_service.recover_interrupted_update()
    try:
        manifest = update_service.fetch_manifest()
        requires_launcher_upgrade = not update_service.launcher_meets_min_version(manifest)
        requires_content_update = update_service.has_update(state, manifest)
        if requires_launcher_upgrade or requires_content_update:
            if args.check_only:
                log.info(
                    "Mandatory update required: installed=%s target=%s launcher=%s required_launcher=%s",
                    state.installed_version,
                    manifest.app_version,
                    update_service.launcher_version(),
                    manifest.min_launcher_version,
                )
                return 2

            log.info(
                "Mandatory update starting: installed=%s target=%s launcher=%s required_launcher=%s",
                state.installed_version,
                manifest.app_version,
                update_service.launcher_version(),
                manifest.min_launcher_version,
            )

            process_service = ProcessService(paths)

            def handoff(checksums: dict[str, str], report_status):
                state.installed_version = manifest.app_version
                state.asset_checksums = checksums
                state.touch_update_time()
                save_install_state(paths.state_dir, state)
                shortcuts.repair_shortcuts()

                report_status("Restarting EzFrames...", "Launching updated runtime.")
                relaunched = process_service.launch_launcher(extra_args=["--log-level", str(args.log_level)])
                report_status("Relaunching updater...", f"Started process pid={relaunched.pid}")

                start = time.time()
                while time.time() - start < 30.0:
                    code = relaunched.poll()
                    if code is None:
                        if time.time() - start >= 5.0:
                            report_status("Updated launcher is running", "Continuing startup.")
                            return
                    elif code == 0:
                        report_status("Updated launcher finished", "Main app handoff complete.")
                        return
                    else:
                        raise RuntimeError(f"Updated launcher exited early with code {code}")
                    time.sleep(0.25)

                report_status("Updated launcher is running", "Continuing startup.")

            state = update_service.apply_update(state, manifest, handoff_callback=handoff)
            save_install_state(paths.state_dir, state)
            shortcuts.repair_shortcuts()
            log.info("Mandatory update complete to %s", state.installed_version)
            return 0

        state.touch_update_time()
        save_install_state(paths.state_dir, state)
    except Exception as exc:
        log.exception("Mandatory update check failed: %s", exc)
        if args.check_only:
            return 2

        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            messagebox.showerror(
                "Update Required",
                "EzFrames could not complete the required update.\n\n"
                "Please relaunch and keep the update window open until it finishes.\n\n"
                f"Details: {exc}",
                parent=root,
            )
            root.destroy()
        except Exception:
            pass
        return 3

    if args.check_only:
        log.info("Check-only complete.")
        return 0

    auth = AuthService(paths, runtime_cfg)
    session = auth.ensure_subscription_interactive()
    if session is None:
        log.warning("Authentication failed or cancelled.")
        return 1

    ticket_path = auth.issue_launch_ticket(session)
    process = ProcessService(paths).launch_main_app(launch_ticket_path=ticket_path)
    log.info("Main app launched with pid=%s (auth=%s)", process.pid, session.source)
    return 0
