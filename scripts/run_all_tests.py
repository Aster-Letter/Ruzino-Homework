#!/usr/bin/env python3
"""
Test Runner Script for Ruzino Project

This script:
1. Recursively finds all 'tests/' folders under './source/'
2. Runs pytest on any 'test_*.py' files found
3. Runs corresponding '*_test.exe' files from './Binaries/Debug/' for any '.cpp' files

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py <test_name>       # Run specific test (e.g., 'rhi_test', 'cpu_slang')
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional





def find_test_directories(source_dir: Path) -> List[Path]:
    """Find all 'tests/' directories under source_dir recursively."""
    test_dirs = []
    for root, dirs, _ in os.walk(source_dir):
        # Skip spdlog tests directory
        if 'spdlog' in root:
            continue
        if 'tests' in dirs:
            test_dirs.append(Path(root) / 'tests')
    return test_dirs


def find_python_test_files(test_dir: Path) -> List[Path]:
    """Find all test_*.py files in the given directory."""
    return list(test_dir.glob('test_*.py'))


def find_cpp_test_files(test_dir: Path) -> List[Path]:
    """Find all *.cpp files in the given directory."""
    return list(test_dir.glob('*.cpp'))


def cpp_to_exe_name(cpp_file: Path) -> str:
    """Convert cpp filename to expected exe name.
    
    For example:
    - some_file.cpp -> some_file_test.exe
    - renderer.cpp -> renderer_test.exe
    """
    base_name = cpp_file.stem
    if base_name.endswith('_test'):
        return f"{base_name}.exe"
    else:
        return f"{base_name}_test.exe"


def should_run_test(exe_name: str, test_filter: Optional[str]) -> bool:
    """Check if a test should be run based on filter."""
    if test_filter is None:
        return True
    
    # Match test name (with or without _test.exe suffix)
    test_base = exe_name.replace('_test.exe', '').replace('.exe', '')
    filter_base = test_filter.replace('_test', '').replace('.exe', '')
    
    return filter_base.lower() in test_base.lower()


def run_pytest(test_dir: Path, test_filter: Optional[str] = None) -> Tuple[int, int, List[str]]:
    """Run pytest in the given directory.
    
    Returns:
        Tuple of (passed, failed, failed_test_names)
    """
    python_tests = find_python_test_files(test_dir)
    
    if not python_tests:
        return 0, 0, []
    
    # Apply filter if specified
    if test_filter:
        python_tests = [t for t in python_tests if test_filter.lower() in t.stem.lower()]
        if not python_tests:
            return 0, 0, []
    
    print(f"\n{'='*80}")
    print(f"Running pytest in: {test_dir}")
    print(f"{'='*80}")
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test_file in python_tests:
        print(f"\n--- Running: {test_file.name} ---")
        
        try:
            result = subprocess.run(
                ['pytest', str(test_file), '-v', '--tb=short'],
                cwd=test_dir,
                timeout=300  # 5 minute timeout per test file
            )
            
            if result.returncode == 0:
                print(f"✓ PASSED: {test_file.name}")
                passed += 1
            else:
                print(f"✗ FAILED: {test_file.name} (exit code: {result.returncode})")
                failed += 1
                failed_tests.append(f"{test_dir.name}/{test_file.name}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {test_file.name}")
            failed += 1
            failed_tests.append(f"{test_dir.name}/{test_file.name} (TIMEOUT)")
        except Exception as e:
            print(f"✗ ERROR: {test_file.name} - {str(e)}")
            failed += 1
            failed_tests.append(f"{test_dir.name}/{test_file.name} (ERROR: {str(e)})")
    
    return passed, failed, failed_tests


def run_cpp_tests(test_dir: Path, binaries_dir: Path, test_filter: Optional[str] = None) -> Tuple[int, int, List[str]]:
    """Run C++ test executables corresponding to cpp files in test_dir.
    
    Returns:
        Tuple of (passed, failed, failed_test_names)
    """
    cpp_files = find_cpp_test_files(test_dir)
    
    if not cpp_files:
        return 0, 0, []
    
    print(f"\n{'='*80}")
    print(f"Running C++ tests from: {test_dir}")
    print(f"{'='*80}")
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for cpp_file in cpp_files:
        exe_name = cpp_to_exe_name(cpp_file)
        
        # Apply filter if specified
        if test_filter and not should_run_test(exe_name, test_filter):
            continue
        
        exe_path = binaries_dir / exe_name
        
        print(f"\n--- Running: {exe_name} (from {cpp_file.name}) ---")
        
        if not exe_path.exists():
            print(f"⚠ SKIPPED: {exe_name} not found in {binaries_dir}")
            continue
        
        try:
            result = subprocess.run(
                [str(exe_path)],
                cwd=binaries_dir,
                timeout=300  # 5 minute timeout per test
            )
            
            if result.returncode == 0:
                print(f"✓ PASSED: {exe_name}")
                passed += 1
            else:
                print(f"✗ FAILED: {exe_name} (exit code: {result.returncode})")
                failed += 1
                failed_tests.append(f"{exe_name}")
                
        except subprocess.TimeoutExpired:
            print(f"✗ TIMEOUT: {exe_name}")
            failed += 1
            failed_tests.append(f"{exe_name} (TIMEOUT)")
        except Exception as e:
            print(f"✗ ERROR: {exe_name} - {str(e)}")
            failed += 1
            failed_tests.append(f"{exe_name} (ERROR: {str(e)})")
    
    return passed, failed, failed_tests


def main():
    """Main test runner."""
    # Parse command line arguments
    test_filter = None
    if len(sys.argv) > 1:
        test_filter = sys.argv[1]
        print(f"Running tests matching: {test_filter}")
    
    # Setup paths - script is now in scripts/ directory, so go up one level to project root
    script_dir = Path(__file__).parent.parent  # Go up from scripts/ to project root
    source_dir = script_dir / 'source'
    binaries_dir = script_dir / 'Binaries' / 'Debug'
    
    print(f"Starting test run...")
    print(f"Searching for tests in: {source_dir}")
    print("-" * 80)
    
    # Find all test directories
    test_dirs = find_test_directories(source_dir)
    print(f"Found {len(test_dirs)} test directories")
    
    # Run all tests
    total_pytest_passed = 0
    total_pytest_failed = 0
    total_cpp_passed = 0
    total_cpp_failed = 0
    all_failed_tests = []
    
    for test_dir in test_dirs:
        print(f"\nProcessing: {test_dir.relative_to(script_dir)}")
        
        # Run Python tests
        pytest_passed, pytest_failed, pytest_failed_tests = run_pytest(test_dir, test_filter)
        total_pytest_passed += pytest_passed
        total_pytest_failed += pytest_failed
        all_failed_tests.extend(pytest_failed_tests)
        
        if pytest_passed + pytest_failed > 0:
            print(f"  Python tests: {pytest_passed} passed, {pytest_failed} failed")
        
        # Run C++ tests
        cpp_passed, cpp_failed, cpp_failed_tests = run_cpp_tests(test_dir, binaries_dir, test_filter)
        total_cpp_passed += cpp_passed
        total_cpp_failed += cpp_failed
        all_failed_tests.extend(cpp_failed_tests)
        
        if cpp_passed + cpp_failed > 0:
            print(f"  C++ tests: {cpp_passed} passed, {cpp_failed} failed")
    
    # Print summary to console
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Python Tests: {total_pytest_passed} passed, {total_pytest_failed} failed")
    print(f"C++ Tests:    {total_cpp_passed} passed, {total_cpp_failed} failed")
    print(f"Overall:      {total_pytest_passed + total_cpp_passed} passed, "
          f"{total_pytest_failed + total_cpp_failed} failed")
    
    # Print failed tests if any
    if all_failed_tests:
        print("\n" + "="*80)
        print("FAILED TESTS:")
        print("="*80)
        for failed_test in all_failed_tests:
            print(f"  ✗ {failed_test}")
    
    print("="*80)
    
    # Exit with error code if any tests failed
    if total_pytest_failed + total_cpp_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
