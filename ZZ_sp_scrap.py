from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import time


def login(user, pw, url):
    # Define the username and password
    username = user
    password = pw

    # Start the bot (using Firefox in this example)
    driver = webdriver.Chrome()

    # Navigate to the login page (replace this with your actual URL)
    driver.get(url)  # Example URL, replace with actual login URL

    try:
        # Wait for the username field to be present and then input the username
        username_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'input28'))
        )
        username_element.send_keys(username)

        # click button with Value 'Next' 
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@value='Next']"))
        )
        next_button.click()

        # Wait for the password field to be present and then input the password
        password_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'input59'))
        )
        password_element.send_keys(password)

        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//input[@value='Sign In']"))
        )
        submit_button.click()


    except TimeoutException:
        print("Element not found within the given time")
    finally:
        # You can return the driver if you need further interactions after login
        return driver


def criteria_selection(driver):


    try:
        dropdown_button_com= WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.dropdown-toggle.selectpicker.btn-default.btn-sm[title='All Commodities']"))
        )

        dropdown_button_com.click()


        checkbox_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "span[data-chk='on'][data-role='checkbox']"))
        )
        checkbox_element.click()


        unselect = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn.dropdown-toggle.btn-default.btn-sm[title='No (Search all Commodities)']"))
        )
        unselect.click()


        select_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='snlInput146']"))
        )
        
        # Create a Select object
        select = Select(select_element)
        
        # Select the option for Copper using its value
        select.select_by_value("5740") 

       

    except TimeoutException:
        print("Element not found within the given time")

    except Exception as e:
        # Catch any other general exceptions
        print(f"An error occurred: {str(e)}")


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
    url = 'https://www.capitaliq.spglobal.com/web/client?auth=inherit#office/screener?perspective=243327' 

    access_path = r'data\int\sp_access.txt'
    user, pw = get_access_data(access_path)

    dr = login(user, pw, url)

    criteria_selection(dr)
