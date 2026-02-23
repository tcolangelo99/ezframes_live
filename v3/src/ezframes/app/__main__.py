import os

from ezframes.app.auth_gate import enforce_authorized_launch
from ezframes.app.legacy_parity import main as legacy_main
from ezframes.app.ui_controller import main as new_main


if __name__ == "__main__":
    if os.environ.get("EZFRAMES_APP_AUTH_OK", "") != "1":
        auth_code = enforce_authorized_launch(interactive_error=True, consume_ticket=True)
        if auth_code != 0:
            raise SystemExit(auth_code)
        os.environ["EZFRAMES_APP_AUTH_OK"] = "1"

    mode = os.environ.get("EZFRAMES_APP_MODE", "new").strip().lower()
    if mode == "legacy":
        raise SystemExit(legacy_main())
    raise SystemExit(new_main())
