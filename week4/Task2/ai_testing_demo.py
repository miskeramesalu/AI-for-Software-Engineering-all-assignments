"""
=== Task 2: AI-Enhanced Software Testing ===
Course: AI in Software Engineering – Week 4

This script demonstrates how AI-assisted testing can help developers quickly
generate automated test cases and element locators. It uses Selenium WebDriver
to test a simple local login page (`local_test_login.html`).

AI Contribution:
- Suggested test case structure and XPath locators.
- Helped generate boundary and negative test cases automatically.

Offline Version:
- Uses a manually downloaded ChromeDriver to avoid online downloads.
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# ==== UPDATE THIS PATH to the location where you saved chromedriver.exe ====
CHROME_DRIVER_PATH = "C:/WebDriver/bin/chromedriver.exe"

def run_login_test(username: str, password: str):
    """Run a single automated login attempt on the local HTML page."""
    html_path = os.path.abspath("local_test_login.html")
    chrome_opts = Options()
    chrome_opts.add_argument("--headless")  # Run without opening a window
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")

    # Use manual driver path (offline)
    driver = webdriver.Chrome(executable_path=CHROME_DRIVER_PATH, options=chrome_opts)
    driver.get(f"file:///{html_path}")

    # Fill username and password fields
    driver.find_element(By.ID, "username").send_keys(username)
    driver.find_element(By.ID, "password").send_keys(password)
    driver.find_element(By.ID, "loginBtn").click()

    # Wait briefly for result to appear
    time.sleep(1)
    result = driver.find_element(By.ID, "result").text
    driver.quit()
    return result


def test_valid_login():
    result = run_login_test("valid_user", "valid_pass")
    assert "Success" in result, f"Expected success but got: {result}"
    print("✅ Valid login test passed.")


def test_invalid_login():
    result = run_login_test("wrong_user", "wrong_pass")
    assert "Invalid" in result, f"Expected invalid message but got: {result}"
    print("✅ Invalid login test passed.")


if __name__ == "__main__":
    print("Running Task 2: AI-Enhanced Software Testing...\n")
    test_valid_login()
    test_invalid_login()
    print("\nAll Task 2 tests executed successfully.")