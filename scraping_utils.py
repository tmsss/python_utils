# from selenium import webdriver
from seleniumwire import webdriver
from bs4 import BeautifulSoup
import os

executable_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'geckodriver.exe'))

def get_driver():
    driver = webdriver.Firefox(executable_path=executable_path)
    driver.implicitly_wait(30)
    return driver


def get_driver_soup(driver, url):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup


def get_soup(url):
    driver = get_driver()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup
