from __future__ import annotations

import argparse
import logging

from ezframes.common.config import AppPaths, RuntimeConfig
from ezframes.common.logging_utils import configure_logging
from ezframes.common.state import load_install_state, save_install_state
from ezframes.launcher.auth_service import AuthService
from ezframes.launcher.migration_service import MigrationService
from ezframes.launcher.process_service import ProcessService
from ezframes.launcher.update_service import UpdateService


log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EzFrames v3 launcher")
    parser.add_argument("--check-only", action="store_true", help="Check migration/auth/update and exit.")
    parser.add_argument("--skip-update", action="store_true", help="Skip update check.")
    parser.add_argument("--log-level", default="INFO", help="Log level.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    paths = AppPaths.default()
    paths.ensure_layout()
    configure_logging(paths.logs_dir, level=args.log_level)

    runtime_cfg = RuntimeConfig.from_env()
    state = load_install_state(paths.state_dir)

    migration = MigrationService(paths)
    state = migration.run_if_needed(state)
    save_install_state(paths.state_dir, state)

    update_service = UpdateService(paths, runtime_cfg)
    if not args.skip_update:
        try:
            manifest = update_service.fetch_manifest()
            if not update_service.launcher_meets_min_version(manifest):
                log.error(
                    "Launcher version too old: current=%s required=%s",
                    update_service.launcher_version(),
                    manifest.min_launcher_version,
                )
                if not args.check_only:
                    update_service.prompt_launcher_upgrade(manifest)
                return 2
            if update_service.has_update(state, manifest):
                if args.check_only:
                    log.info("Update available: %s -> %s", state.installed_version, manifest.app_version)
                else:
                    if update_service.prompt_update(state.installed_version, manifest.app_version):
                        state = update_service.apply_update(state, manifest)
                        save_install_state(paths.state_dir, state)
                        log.info("Update complete to %s", state.installed_version)
            else:
                state.touch_update_time()
                save_install_state(paths.state_dir, state)
        except Exception as exc:
            log.exception("Update check failed: %s", exc)

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
