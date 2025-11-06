"""
Test Runner Script
Run all tests and generate coverage report
"""
import subprocess
import sys
import os


def run_tests():
    """Run all tests with pytest"""
    print("=" * 60)
    print("AutoNLP-Agent Test Suite")
    print("=" * 60)
    
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests with pytest
    print("\n🧪 Running tests...\n")
    
    test_commands = [
        # Run core tests
        ["pytest", "tests/test_core.py", "-v", "--tb=short"],
        
        # Run API integration tests
        ["pytest", "tests/test_api_integration.py", "-v", "--tb=short"],
        
        # Run upload tests
        ["pytest", "tests/test_upload.py", "-v", "--tb=short"],
        
        # Run all tests with coverage
        ["pytest", "tests/", "-v", "--cov=app", "--cov-report=html", "--cov-report=term"],
    ]
    
    for i, cmd in enumerate(test_commands[:-1], 1):  # Skip coverage command initially
        print(f"\n{'='*60}")
        print(f"Test Suite {i}/{len(test_commands)-1}: {' '.join(cmd)}")
        print('='*60 + "\n")
        
        try:
            result = subprocess.run(cmd, check=False)
            if result.returncode != 0:
                print(f"\n⚠️  Some tests failed in suite {i}")
        except FileNotFoundError:
            print(f"❌ pytest not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
            result = subprocess.run(cmd, check=False)
    
    # Run coverage report
    print(f"\n{'='*60}")
    print("Generating Coverage Report")
    print('='*60 + "\n")
    
    try:
        subprocess.run(test_commands[-1], check=False)
        print("\n✅ Coverage report generated in htmlcov/index.html")
    except:
        print("\n⚠️  Coverage report generation skipped")
    
    print("\n" + "="*60)
    print("✅ Test execution completed!")
    print("="*60)


if __name__ == "__main__":
    run_tests()
