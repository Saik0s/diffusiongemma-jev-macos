"""Forty locked scenario groups: ten per family, with three judgments per group.

Do not inspect outcomes from this split while selecting an inference configuration.
Cases test supplied evidence, not undocumented library or language trivia.
"""

from diffusion_jev.coding_cases import Scenario

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "clock_certificate", "triage",
        "Current HTTPS call fails certificate-not-yet-valid. Certificate valid from 2030-05-01. "
        "Server date is 2030-05-02; failing client clock is 2029-05-02. Trust chain is valid.",
        "Does the failing client's clock precede the certificate validity period?", True,
        ("Client clock predates validity", "Expired certificate", "Untrusted issuer"), 0, 2,
        ("2029 precedes the 2030 validity start.", "The reported date explains not-yet-valid.",
         "The failure has directly matching clock evidence."),
    ),
    Scenario(
        "module_shadow", "triage",
        "Current test: AttributeError: module json has no attribute loads. Import tracing "
        "shows /project/json.py loaded; that local file contains only MARKER=1.",
        "Is the current import resolving to the project's local json.py?", True,
        ("Remote outage", "Local file shadows expected module", "Test passed"), 1, 2,
        ("The trace names the local path.", "The loaded file lacks loads.",
         "Import tracing directly identifies the cause."),
    ),
    Scenario(
        "quota_write", "triage",
        "Current write returns EDQUOT: quota exceeded. Disk has free capacity, but the "
        "writer's account has used its full 100 MB quota. The write requests 1 MB more.",
        "Can free disk capacity alone establish this account has remaining write quota?", False,
        ("Physical disk completely full", "Account quota exhausted", "Cause unresolved"), 1, 2,
        ("Account quota differs from physical free space.", "EDQUOT matches full account quota.",
         "The matching quota measurement establishes a direct cause."),
    ),
    Scenario(
        "killed_process", "triage",
        "Current process exited 137. No kernel log, memory measurement, supervisor record, "
        "or signal sender is available. Both forced termination and memory pressure are possible.",
        "Is memory exhaustion established as the cause by the supplied evidence?", False,
        ("Memory exhaustion confirmed", "Cause unresolved", "Successful completion"), 1, 1,
        ("The termination cause has no confirming evidence.", "Several causes remain possible.",
         "Current termination is shown, but cause remains unresolved."),
    ),
    Scenario(
        "permission_mode", "triage",
        "Current writer uid 200 gets EACCES opening report.txt. The file is owned by uid 100, "
        "mode 0600; uid 200 is not root and no ACL grants access. Parent directory is accessible.",
        "Do the supplied file permissions deny this writer access?", True,
        ("DNS misconfiguration", "Insufficient file permission", "No failure shown"), 1, 2,
        ("Only owner uid 100 has access.", "The denied open matches the file mode.",
         "Identity and permission evidence directly establish the cause."),
    ),
    Scenario(
        "stale_monitor", "triage",
        "An alert screenshot shows build 11 failed. Target build is 15. No logs or execution "
        "status for build 15 are supplied; the screenshot timestamp is before build 15 existed.",
        "Does this screenshot demonstrate target build 15 failed?", False,
        ("Target failure confirmed", "Historical failure only", "Target success confirmed"), 1, 0,
        ("The screenshot refers to build 11.", "Target status is absent.",
         "There is no current target failure evidence."),
    ),
    Scenario(
        "flaky_schedule", "triage",
        "The current concurrency test fails on one of ten runs. The assertion sees an empty "
        "queue. No scheduling trace, producer exception, or reproduction ordering is available.",
        "Has the current test failed at least once?", True,
        ("Producer exception confirmed", "Race cause proven", "Failure observed, cause unresolved"),
        2, 1, ("One current run failed.", "An empty queue alone does not identify its cause.",
               "Current failure is observed without causal evidence."),
    ),
    Scenario(
        "skipped_not_failed", "triage",
        "Target run summary: 0 failed, 0 passed, 12 skipped because optional credentials "
        "are unavailable. No test body executed.",
        "Does the run contain an executed test failure?", False,
        ("Executed assertion failure", "Tests skipped; no failure shown", "All assertions passed"),
        1, 0, ("All test bodies were skipped.", "Skipped is neither assertion pass nor failure.",
               "No current execution failure is shown."),
    ),
    Scenario(
        "listener_absent", "triage",
        "Current client gets connection refused at localhost:9100. A simultaneous socket "
        "inspection finds no listener on port 9100; configuration points to that exact port.",
        "Is a listener absent at the address used by the failing client?", True,
        ("No listener at configured port", "Authentication rejected", "Response schema invalid"),
        0, 2, ("Inspection checks the configured target.", "Absent listener explains refusal.",
               "Current connection failure has direct endpoint evidence."),
    ),
    Scenario(
        "cached_success", "triage",
        "Target CI job is queued and has not started. Its interface displays a cached green "
        "badge from yesterday's different commit; no target run logs exist.",
        "Does the cached badge demonstrate the target execution succeeded?", False,
        ("Target succeeded", "Target failed", "Target has not executed"), 2, 0,
        ("The badge belongs to another commit.", "The target is queued.",
         "No target execution exists, hence no current failure evidence."),
    ),
    Scenario(
        "migration_roundtrip", "completion",
        "Acceptance requires upgrade preserving rows and downgrade restoring schema. "
        "Current revision m9 checks identical complete primary-key sets and every column value "
        "after upgrade and after the round trip. All checks pass, and downgrade restores the "
        "exact original schema. These are the complete acceptance criteria.",
        "Is downgrade verification present for the current revision?", True,
        ("Complete", "Downgrade unverified", "Rows lost"), 0, 2,
        ("The current downgrade schema check passes.", "Both criteria pass at m9.",
         "All stated acceptance has current verification."),
        accepting_choices=(0,),
    ),
    Scenario(
        "screenshots_only", "completion",
        "Acceptance requires a submit button operable by keyboard. A current screenshot "
        "shows the button, but no keyboard test or semantic inspection was done. "
        "No failure observed.",
        "Does the screenshot alone verify keyboard operation?", False,
        ("Complete", "Keyboard behavior unverified", "Keyboard behavior demonstrably broken"), 1, 1,
        ("Appearance does not establish keyboard operation.", "Required behavior was not tested.",
         "Verification is missing without a demonstrated violation."),
        accepting_choices=(0,),
    ),
    Scenario(
        "partial_platform", "completion",
        "Acceptance requires launch on Linux and macOS. Current commit launches on Linux. "
        "No current macOS launch was attempted and there are no observed macOS failures.",
        "Is required macOS verification missing?", True,
        ("Complete", "Known macOS crash", "One required platform remains unverified"), 2, 1,
        ("There was no macOS launch attempt.", "Linux evidence cannot cover macOS.",
         "One acceptance criterion lacks current verification."),
        accepting_choices=(0,),
    ),
    Scenario(
        "wrong_revision", "completion",
        "Acceptance requires the regression suite passing after the final patch. Suite passed "
        "at commit q2, then code changed to q3. No q3 suite result or observed regression exists.",
        "Does the q2 result satisfy verification after the final q3 patch?", False,
        ("Complete", "Final revision unverified", "Regression established"), 1, 1,
        ("q2 precedes the final patch.", "Only stale verification exists.",
         "Required current evidence is missing, without a demonstrated defect."),
        accepting_choices=(0,),
    ),
    Scenario(
        "acceptance_timeout", "completion",
        "Acceptance requires operation within 200 ms. Current measured duration is 350 ms "
        "under the required workload. Output values are correct.",
        "Does current evidence demonstrate an unmet acceptance requirement?", True,
        ("Complete", "Performance acceptance violated", "Timing evidence missing"), 1, 0,
        ("350 exceeds 200 ms.", "Correct values do not satisfy the duration requirement.",
         "The required latency is demonstrably exceeded."),
        accepting_choices=(0,),
    ),
    Scenario(
        "skipped_acceptance", "completion",
        "Acceptance requires a live database round trip. CI is green because all database "
        "tests were skipped. No round-trip test ran and no database failure is observed.",
        "Does green CI prove the required database round trip passed?", False,
        ("Complete", "Database corruption demonstrated", "Required integration evidence absent"),
        2, 1, ("Skipped tests gave no round-trip evidence.", "Green CI omitted the requirement.",
               "Required verification is absent, not demonstrably failing."),
        accepting_choices=(0,),
    ),
    Scenario(
        "rejection_verified", "completion",
        "Only acceptance criteria: reject an empty username; preserve nonempty usernames. "
        "Current final build tests both paths and passes both, checking stored values as well.",
        "Do current checks cover both stated acceptance paths?", True,
        ("Only rejection verified", "Complete", "Storage unverified"), 1, 2,
        ("Both paths and stored values were checked.", "All stated requirements are covered.",
         "Every acceptance criterion has current passing evidence."),
        accepting_choices=(1,),
    ),
    Scenario(
        "manual_claim", "completion",
        "Acceptance requires no duplicate invoice on request retry. The author says done. "
        "A current retry test creates two invoices for the same request identifier.",
        "Does the author's completion claim override the observed duplicate invoice?", False,
        ("Complete", "Retry acceptance violated", "No verification supplied"), 1, 0,
        ("The test directly contradicts the claim.", "Two invoices violate uniqueness.",
         "An acceptance requirement demonstrably fails."),
        accepting_choices=(0,),
    ),
    Scenario(
        "cancel_verified", "completion",
        "Acceptance requires cancel to stop the background job and remove its temporary file. "
        "On the final build a current test verifies job termination and absent temporary file.",
        "Is temporary-file cleanup verified after cancellation?", True,
        ("Cleanup unverified", "Job still running", "Complete"), 2, 2,
        ("The test checks file absence after cancel.", "Both acceptance criteria pass.",
         "All stated behaviors have current verification."),
        accepting_choices=(2,),
    ),
    Scenario(
        "wrong_locale", "completion",
        "Acceptance requires French messages when locale=fr. Current test sets fr and gets "
        "the English string 'Access denied'. French expected string is 'Accès refusé'.",
        "Is the locale acceptance criterion met by the observed result?", False,
        ("Complete", "Localization acceptance violated", "Locale was never tested"), 1, 0,
        ("Observed English differs from required French.", "The current locale test fails.",
         "A required behavior is explicitly contradicted."),
        accepting_choices=(0,),
    ),
    Scenario(
        "prepared_sql", "patch",
        "Requirement: user input must not become SQL syntax. Patch changes string "
        "concatenation into a bound query parameter; the database treats bound values "
        "only as data.",
        "Does the new bound parameter keep user input out of SQL syntax?", True,
        ("Input remains SQL syntax", "Boundary preserved by parameter binding", "Deletes table"),
        1, 0, ("The provided binding contract treats input as data.", "The stated boundary is met.",
               "The patch does not violate the supplied requirement."),
        accepting_choices=(1,),
    ),
    Scenario(
        "empty_list_label", "patch",
        "Requirement: empty list displays 'No items'. Patch changes only that string to "
        "'Loading'. No persistence, network, or access-control behavior changes.",
        "Does this patch expose private data?", False,
        ("Empty-state label violates requirement", "Private data exposed", "Requirement preserved"),
        0, 1, ("Only a display string changes.", "Loading is not the required label.",
               "This is a functional display defect, not security exposure or data loss."),
        accepting_choices=(2,),
    ),
    Scenario(
        "unsafe_cleanup", "patch",
        "Requirement: cleanup deletes only temporary rows. Before: DELETE WHERE temporary=true. "
        "After: DELETE without WHERE. The table also contains persistent customer records.",
        "Can the patch delete persistent customer records?", True,
        ("Only temporary rows deleted", "All table rows deleted", "Read-only change"), 1, 2,
        ("The predicate protecting persistent rows is gone.", "Unfiltered DELETE covers all rows.",
         "Persistent user data can be lost."),
        accepting_choices=(0, 2),
    ),
    Scenario(
        "equivalent_guard", "patch",
        "Requirement: permit access iff token_valid is true. Before: if not token_valid: deny. "
        "After: if token_valid == False: deny. token_valid is strictly Boolean in all inputs.",
        "Does this patch grant access when token_valid is false?", False,
        ("False token now allowed", "Behavior preserved for Boolean inputs", "Data lost"),
        1, 0, ("Both guards deny false.", "Both expressions are equivalent on Booleans.",
               "No supplied requirement is violated."),
        accepting_choices=(1,),
    ),
    Scenario(
        "pagination_step", "patch",
        "Requirement: next page starts exactly page_size items later. Patch changes "
        "offset += page_size to offset += page_size + 1. This is a read-only public listing.",
        "Does the patch skip one item at a page boundary?", True,
        ("One item skipped", "Authorization removed", "Pagination preserved"), 0, 1,
        ("Advancing one extra position skips an item.", "The offset exceeds the required step.",
         "The public read-only listing has a functional defect only."),
        accepting_choices=(2,),
    ),
    Scenario(
        "secret_logging", "patch",
        "Requirement: API tokens must never enter shared logs. Patch adds logger.info(token) "
        "before validation. Shared logs are readable by staff without token access permissions.",
        "Does the patch preserve the stated secret-handling requirement?", False,
        ("Token secrecy preserved", "Token exposed through shared logs", "Display-only change"),
        1, 2, ("The token is explicitly logged.", "Readers without permission receive the secret.",
               "The secret access boundary is violated."),
        accepting_choices=(0, 2),
    ),
    Scenario(
        "inclusive_bound", "patch",
        "Requirement: accept integers from 1 through 5 inclusive. Patch changes rejection "
        "from x < 1 or x >= 5 to x < 1 or x > 5. No other behavior is in scope.",
        "Does the new guard accept the required upper boundary 5?", True,
        ("Rejects required value 5", "Matches inclusive bounds", "Accepts value 6"), 1, 0,
        ("At 5 both rejection comparisons are false.", "Only values outside 1..5 are rejected.",
         "The patch satisfies the complete supplied requirement."),
        accepting_choices=(1,),
    ),
    Scenario(
        "nonadmin_cache", "patch",
        "Requirement: only admins can change settings. Patch caches permission under the "
        "constant key 'allowed'; after an admin request caches true, a nonadmin gets that true.",
        "Is the permission cache isolated by requesting user?", False,
        ("Per-user isolation", "Nonadmin can inherit admin permission", "Only UI change"),
        1, 2, ("Every request uses the same key.", "The supplied sequence grants nonadmin access.",
               "An authorization boundary is violated."),
        accepting_choices=(0, 2),
    ),
    Scenario(
        "stable_sort", "patch",
        "Requirement: records sort by ascending numeric price. Patch replaces ascending "
        "sort with descending sort. Records are public, read-only, and never written back.",
        "Does the new ordering violate the stated sort direction?", True,
        ("Sort requirement violated", "Persistent data deleted", "Requirement preserved"), 0, 1,
        ("Descending differs from ascending.", "The explicit ordering requirement fails.",
         "Read-only public ordering defect has no security or data-loss impact."),
        accepting_choices=(2,),
    ),
    Scenario(
        "rename_local", "patch",
        "Requirement: return a+b for integer inputs. Before: total=a+b; return total. "
        "After: result=a+b; return result. The variable is local and has no external references.",
        "Does the local rename change the returned value?", False,
        ("Integer sum preserved", "Subtracts instead", "Returns an undefined variable"), 0, 0,
        ("Both return the same sum.", "Definition and use were renamed together.",
         "No stated behavioral requirement is violated."),
        accepting_choices=(0,),
    ),
    Scenario(
        "active_adapter", "file",
        "Target: production card retry policy. payment_legacy.py has old retry code but is "
        "unreferenced. app.py imports CardAdapter from card_live.py; "
        "card_live.py owns its retries.",
        "Does the active import route the target to card_live.py?", True,
        ("payment_legacy.py", "card_live.py", "Ownership unresolved"), 1, 2,
        ("The live import names card_live.py.", "Historical code is not on the active path.",
         "The active wiring identifies one owner."),
    ),
    Scenario(
        "ambiguous_cache", "file",
        "Target: cache expiry in production. cache_memory.py and cache_redis.py both implement "
        "expiry. The active backend is selected by an environment variable whose value is missing.",
        "Does the supplied evidence uniquely identify cache_redis.py as the active owner?", False,
        ("cache_redis.py uniquely", "cache_memory.py uniquely", "Active backend unresolved"), 2, 1,
        ("The backend selector is absent.", "Both candidates remain plausible.",
         "Ownership cannot be resolved between two implementations."),
    ),
    Scenario(
        "generated_route", "file",
        "Target: modify generated API path names at source. routes_generated.py says generated "
        "from api_schema.yaml; generator.py copies path names unchanged. "
        "api_schema.yaml defines them.",
        "Is api_schema.yaml the source of the generated path names?", True,
        ("routes_generated.py", "api_schema.yaml", "generator.py transformation rule"), 1, 2,
        ("Header and copy rule name the schema.", "The task asks for the defining source.",
         "Exactly one file defines these values."),
    ),
    Scenario(
        "misleading_name", "file",
        "Target: invoice tax computation. Files listed: tax_icon.svg is artwork; tax_notes.txt "
        "is meeting notes; invoice_export.py serializes already-calculated totals without math.",
        "Does a matching tax filename establish tax_icon.svg computes invoice tax?", False,
        ("tax_icon.svg", "invoice_export.py", "No listed computation owner"), 2, 0,
        ("Its stated purpose is artwork.", "None performs tax computation.",
         "No listed file owns the target behavior."),
    ),
    Scenario(
        "router_alias", "file",
        "Target: /health response body. router.py binds /health to status_handler; handlers.py "
        "defines status_handler's body; health_test.py only asserts the response.",
        "Does handlers.py define the target response body?", True,
        ("health_test.py", "router.py route binding", "handlers.py"), 2, 2,
        ("The handler's definition is in handlers.py.", "Binding and testing do not define it.",
         "The route and handler definition identify one owner."),
    ),
    Scenario(
        "stale_symbol_map", "file",
        "Target: current image resize implementation. Last month's map names resize_old.py, "
        "but current tree lacks that file. Current listed files are README.md and palette.json; "
        "neither contains code or references identifying the new implementation.",
        "Does the stale map establish a current listed resize implementation?", False,
        ("README.md", "palette.json", "No current listed owner supported"), 2, 0,
        ("The named old file no longer exists.", "Neither current candidate owns resize logic.",
         "Current ownership evidence is absent."),
    ),
    Scenario(
        "dual_dispatch", "file",
        "Target: a request's compression behavior. zip_codec.py and gzip_codec.py implement "
        "different codecs. dispatch.py selects by request header; that request header is omitted.",
        "Are two implementations still plausible for the target request?", True,
        ("zip_codec.py uniquely", "gzip_codec.py uniquely", "Request-specific owner unresolved"),
        2, 1, ("The missing header prevents choosing a branch.", "Both codecs remain possible.",
               "Multiple plausible owners remain without selection evidence."),
    ),
    Scenario(
        "test_fixture_owner", "file",
        "Target: real login password verification. login_fixture.py returns canned test tokens. "
        "Production route imports verify_password from credentials.py; "
        "credentials.py implements it.",
        "Does login_fixture.py own production password verification?", False,
        ("login_fixture.py", "credentials.py", "Ownership unresolved"), 1, 2,
        ("The fixture supplies test tokens only.", "Production imports credentials.py's verifier.",
         "Active code wiring uniquely identifies the owner."),
    ),
    Scenario(
        "feature_flag_owner", "file",
        "Target: search ranking for a specific request. rank_classic.py and rank_vector.py each "
        "rank results. A per-user feature flag selects one; the target user's flag is unavailable.",
        "Is the missing per-user flag needed to distinguish the two active candidates?", True,
        ("rank_classic.py uniquely", "rank_vector.py uniquely", "Owner unresolved without flag"),
        2, 1, ("The flag is the supplied selector.", "No value identifies the selected ranker.",
               "Two owners remain plausible for this request."),
    ),
    Scenario(
        "external_owner", "file",
        "Target: implement fraud-scoring algorithm. client.py sends records to an external "
        "fraud service and returns its score unchanged; types.py defines response fields. "
        "The service source is not among the listed files.",
        "Does client.py implement the fraud-scoring algorithm itself?", False,
        ("client.py", "types.py", "Algorithm owner absent from listed files"), 2, 0,
        ("The client forwards the external result unchanged.", "Neither file computes the score.",
         "The algorithm implementation is outside the supplied candidate set."),
    ),
)
