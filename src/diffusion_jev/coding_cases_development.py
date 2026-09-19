"""Ten development scenario groups; thirty correlated, public judgments."""

from diffusion_jev.coding_cases import Scenario

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "missing_codec", "triage",
        "Current image test aborts on import: ModuleNotFoundError: pngcodec. The active "
        "environment's installed-package listing excludes pngcodec, a declared dependency.",
        "Is a required package missing from the active environment?", True,
        ("Missing declared dependency", "Image assertion mismatch", "Unresolved cause"), 0, 2,
        ("Installed listing excludes the imported dependency.", "Import and listing agree.",
         "The current failure has a directly identified cause."),
    ),
    Scenario(
        "sporadic_exit", "triage",
        "The current worker exited with status 1. Logs are empty; no exception, resource "
        "measurement, or reproduction is supplied.",
        "Does the evidence establish that memory exhaustion caused the exit?", False,
        ("Memory exhaustion confirmed", "DNS failure confirmed", "Cause unresolved"), 2, 1,
        ("Exit status alone does not identify memory exhaustion.", "No cause evidence exists.",
         "Failure is current but undiagnosed."),
    ),
    Scenario(
        "archived_failure", "triage",
        "The displayed failing log belongs to commit old7. The target is commit new8. "
        "There is no execution result for new8.",
        "Is the displayed failure from a different commit than the target?", True,
        ("Current target failure established", "Only historical failure shown", "Target passed"),
        1, 0, ("old7 differs from new8.", "Only old7 was executed.",
               "No evidence of failure in the target execution is provided."),
    ),
    Scenario(
        "untested_upgrade", "completion",
        "Acceptance requires the upgraded serializer to pass the compatibility suite. "
        "The code was updated; the suite has not run. No observed incompatibility is supplied.",
        "Has the required compatibility verification passed?", False,
        ("Complete", "Implementation present, verification missing", "Known data corruption"),
        1, 1, ("The suite has not run.", "Implementation alone cannot satisfy verification.",
               "There is no observed violation, but required verification is missing."),
        accepting_choices=(0,),
    ),
    Scenario(
        "verified_limits", "completion",
        "Acceptance: reject size above 20 and allow size 20. Current commit r4 tests show "
        "size 21 rejected and size 20 accepted. These are the only acceptance requirements.",
        "Are all stated acceptance requirements verified at the current commit?", True,
        ("Complete", "Boundary requirement failing", "Only stale evidence"), 0, 2,
        ("Both requirements have passing r4 tests.", "All listed acceptance is satisfied.",
         "Current verification covers every stated requirement."),
        accepting_choices=(0,),
    ),
    Scenario(
        "silent_export", "completion",
        "Acceptance requires exported CSV to include a header. A current integration test "
        "reads the exported file and reports missing header. Unit tests elsewhere pass.",
        "Do unrelated passing unit tests establish that export acceptance is satisfied?", False,
        ("Complete", "Acceptance demonstrably failing", "Verification absent"), 1, 0,
        ("The export test directly contradicts acceptance.", "The header is absent.",
         "A required behavior is demonstrably unmet."),
        accepting_choices=(0,),
    ),
    Scenario(
        "cross_tenant_query", "patch",
        "Requirement: queries return only rows for the caller's tenant. Before: WHERE "
        "tenant_id = caller.tenant. After: the tenant predicate is removed; no other filter.",
        "Can the patch expose rows belonging to other tenants?", True,
        ("Tenant isolation preserved", "Only formatting changes", "Tenant boundary removed"),
        2, 2, ("Rows from all tenants become eligible.", "The only tenant filter was removed.",
               "This violates an explicit security boundary."),
        accepting_choices=(0, 1),
    ),
    Scenario(
        "button_caption", "patch",
        "Requirement: button caption must be 'Save'. Patch changes the caption to 'Store'. "
        "The click handler, authorization, and persistence logic remain identical.",
        "Does the supplied patch change authorization behavior?", False,
        ("Authorization bypass", "Caption requirement violated", "All requirements preserved"),
        1, 1, ("Only caption text changes.", "Store differs from required Save.",
               "A functional display requirement fails without security or data-loss impact."),
        accepting_choices=(2,),
    ),
    Scenario(
        "smtp_owner", "file",
        "Target: SMTP retry delay. Files: mail_transport.py implements SMTP retries; "
        "mail_template.py formats email HTML; retry_docs.md describes historical defaults.",
        "Does mail_transport.py own the target retry behavior?", True,
        ("mail_template.py", "mail_transport.py", "Ownership unresolved"), 1, 2,
        ("Its explicit responsibility is SMTP retries.", "HTML and docs do not own the delay.",
         "One listed file uniquely owns the target behavior."),
    ),
    Scenario(
        "hidden_upload_route", "file",
        "Target: validate an upload's MIME type. Listed files are upload_icon.svg (artwork), "
        "download.py (download handler), and upload_notes.md (meeting notes). "
        "No upload code shown.",
        "Does the upload filename alone prove upload_icon.svg owns MIME validation?", False,
        ("upload_icon.svg", "download.py", "No listed owner supported"), 2, 0,
        ("Artwork is not validation code.", "No listed file implements uploads.",
         "No candidate has evidence of owning the behavior."),
    ),
)
