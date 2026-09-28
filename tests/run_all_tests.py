"""
Master Test Runner.
Executes all automated test suites across the project trajcetory (Phases 3, 4, 5, 6, 7).
"""

import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_phase3 import run_phase3_validation
from tests.test_phase4 import run_phase4_validation
from tests.test_phase5 import evaluate_model
from tests.test_phase6 import run_phase6_tests
from tests.test_phase7 import run_phase7_validation

def run_all():
    print("==================================================")
    print("RUNNING COMPLETE MASTER TEST SUITE (PHASES 3 - 7)")
    print("==================================================")

    suite_results = {}

    # Phase 3
    print("\n[1/5] TEST SUITE: PHASE 3 — DATA PREPARATION")
    try:
        run_phase3_validation()
        suite_results["Phase 3: Data Preparation"] = "PASSED"
    except Exception as e:
        print(f"FAILED: {e}")
        suite_results["Phase 3: Data Preparation"] = f"FAILED: {e}"

    # Phase 4
    print("\n[2/5] TEST SUITE: PHASE 4 — DEEP LEARNING MODEL")
    try:
        run_phase4_validation()
        suite_results["Phase 4: Neural Model"] = "PASSED"
    except Exception as e:
        print(f"FAILED: {e}")
        suite_results["Phase 4: Neural Model"] = f"FAILED: {e}"

    # Phase 5
    print("\n[3/5] TEST SUITE: PHASE 5 — MODEL EVALUATION")
    try:
        evaluate_model()
        suite_results["Phase 5: Model Evaluation"] = "PASSED"
    except Exception as e:
        print(f"FAILED: {e}")
        suite_results["Phase 5: Model Evaluation"] = f"FAILED: {e}"

    # Phase 6
    print("\n[4/5] TEST SUITE: PHASE 6 — THRESHOLD-AWARE ENGINE")
    try:
        run_phase6_tests()
        suite_results["Phase 6: Threshold Engine"] = "PASSED"
    except Exception as e:
        print(f"FAILED: {e}")
        suite_results["Phase 6: Threshold Engine"] = f"FAILED: {e}"

    # Phase 7
    print("\n[5/5] TEST SUITE: PHASE 7 — STREAMLIT APPLICATION")
    try:
        run_phase7_validation()
        suite_results["Phase 7: Streamlit Integration"] = "PASSED"
    except Exception as e:
        print(f"FAILED: {e}")
        suite_results["Phase 7: Streamlit Integration"] = f"FAILED: {e}"

    print("\n==================================================")
    print("MASTER TEST SUITE EXECUTION SUMMARY")
    print("==================================================")
    all_passed = True
    for suite, status in suite_results.items():
        print(f" - {suite:35s}: {status}")
        if status != "PASSED":
            all_passed = False

    print("--------------------------------------------------")
    if all_passed:
        print("ALL 5 TEST SUITES PASSED SUCCESSFULLY! MVP IS STABLE.")
    else:
        print("SOME TEST SUITES FAILED! PLEASE CHECK ERRORS ABOVE.")
        sys.exit(1)

if __name__ == "__main__":
    run_all()
