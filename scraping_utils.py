from selenium import webdriver
from bs4 import BeautifulSoup


def get_driver():
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    return driver


def get_driver_soup(driver, url):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    return soup
