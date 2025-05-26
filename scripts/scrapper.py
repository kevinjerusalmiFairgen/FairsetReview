import pandas as pd
import time
import re
import streamlit as st
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
import selenium.common.exceptions

EMAIL = "inspector@fairgen.ai"
PASSWORD = "Inspector123!"

def wait_for_xpath(driver, xpath, timeout=30, poll_frequency=1.0):
    """Custom wait loop for slow-loading XPath elements."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        try:
            elem = driver.find_element(By.XPATH, xpath)
            if elem.is_displayed():
                return elem
        except selenium.common.exceptions.NoSuchElementException:
            pass
        time.sleep(poll_frequency)
    raise TimeoutError(f"Timeout: Element with XPath not found: {xpath}")

def scrap_boostresults(project_url):
    # Headless Firefox setup
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )
    wait = WebDriverWait(driver, 15)

    # Streamlit placeholders
    progress_bar = st.empty()
    status_text = st.empty()
    df_placeholder = st.empty()

    df_live = pd.DataFrame(columns=[
        "Niche", "Niche Size", "Penetration (%)", "Boost MAE (%)", "Training MAE (%)"
    ])

    def ordinal(n):
        return f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"

    try:
        driver.get(project_url)

        # Login
        email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
        password_field = driver.find_element(By.NAME, "password")
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        email_field.send_keys(EMAIL)
        password_field.send_keys(PASSWORD)
        login_button.click()

        # Wait for task list
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_task_qdg41_28")))
        time.sleep(2)
        all_tasks = driver.find_elements(By.CLASS_NAME, "_task_qdg41_28")
        boost_tasks = [t for t in all_tasks if "_boostTask_" in t.get_attribute("class")]
        total = len(boost_tasks)

        for idx, task in enumerate(boost_tasks, 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", task)
                task.click()
                time.sleep(1)

                # Get niche info
                try:
                    nich_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_conditions_h506w_88")))
                    nich_size_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_nicheSize_h506w_96")))
                    nich_text = nich_element.text.strip()
                    nich_size_text = nich_size_element.text.strip()
                except:
                    nich_text = "Not Found"
                    nich_size_text = "Not Found"

                # Click Boost tab
                try:
                    boost_tab = driver.find_element(By.XPATH, '//span[text()="Boost"]')
                    driver.execute_script("arguments[0].click();", boost_tab)
                    time.sleep(1)
                except:
                    st.warning(f"{ordinal(idx)}: Boost tab not found.")
                    continue

                # Wait for metrics container
                try:
                    metric_block_xpath = "/html/body/div/div/div/main/div/div[2]/div[2]/div[2]/div/div[2]/div[3]/div[1]/div[1]/div[1]"
                    container = wait_for_xpath(driver, metric_block_xpath, timeout=25)
                    spans = container.find_elements(By.XPATH, ".//span")
                except Exception as e:
                    st.warning(f"{ordinal(idx)}: Could not locate metric block. {e}")
                    with st.expander(f"HTML dump for boost {idx}"):
                        st.code(driver.page_source[:2000])
                    continue

                # Extract values
                if len(spans) >= 2:
                    boost = float(spans[0].text.strip().split('%')[0])
                    training = float(spans[1].text.strip().split('%')[0])
                else:
                    boost = training = "Not Found"

                clean_niche = nich_text.replace("\n", " ") if nich_text != "Not Found" else "Not Found"
                niche_size = re.search(r"(\d+)", nich_size_text)
                penetration = re.search(r"\(([^)]+)\)", nich_size_text)
                niche_size = niche_size.group(1) if niche_size else "Not Found"
                penetration = penetration.group(1) if penetration else "Not Found"

                df_live.loc[len(df_live)] = {
                    "Niche": clean_niche,
                    "Niche Size": niche_size,
                    "Penetration (%)": penetration,
                    "Boost MAE (%)": boost,
                    "Training MAE (%)": training
                }

                progress_bar.progress(idx / total)
                status_text.text(f"Processed {idx} of {total} boosts")
                df_placeholder.dataframe(df_live)

            except Exception as e:
                st.error(f"{ordinal(idx)} boost failed.")
                st.code(traceback.format_exc(), language="python")
                continue

        return df_live

    except Exception as e:
        st.error(f"Main error: {e}")
        st.code(traceback.format_exc(), language="python")
        return df_live

    finally:
        driver.quit()
