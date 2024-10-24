from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException


def login(user, pw, url):
    # Define the username and password
    username = user
    password = pw

    # Start the bot (using Firefox in this example)
    driver = webdriver.Firefox()

    # Navigate to the login page (replace this with your actual URL)
    driver.get(url)  # Example URL, replace with actual login URL

    try:
        # Wait for the username field to be present and then input the username
        username_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'username'))
        )
        username_element.send_keys(username)

        # Wait for the password field to be present and then input the password
        password_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, 'password'))
        )
        password_element.send_keys(password)

        # Wait for the login button and click it
        login_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="loginButton"]'))
        )
        login_button.click()

    except TimeoutException:
        print("Element not found within the given time")
    finally:
        # You can return the driver if you need further interactions after login
        return driver

def get_access_data(path):
    '''
    Get the username, password, and URL from the access file
    '''
    with open(path, 'r') as f:
        lines = f.readlines()
        user = lines[0].strip()
        pw = lines[1].strip()
    
    return user, pw

if __name__ == '__main__':

    access_path = r'data\int\sp_access.txt'
    user, pw = get_access_data(access_path)

    bot = login(user, pw)
