import re
import time
import traceback

import pandas as pd
import streamlit as st

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager import firefox
GeckoDriverManager = firefox.GeckoDriverManager

EMAIL = "inspector@fairgen.ai"
PASSWORD = "Inspector123!"

def scrap_boostresults(project_url):
    # Setup headless Firefox for Streamlit Cloud
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")

    driver = webdriver.Firefox(
        service=Service(GeckoDriverManager().install()),
        options=options
    )
    wait = WebDriverWait(driver, 20)

    # Streamlit layout
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

        email_field = wait.until(EC.presence_of_element_located((By.NAME, "email")))
        password_field = driver.find_element(By.NAME, "password")
        login_button = driver.find_element(By.XPATH, "//button[@type='submit']")
        email_field.send_keys(EMAIL)
        password_field.send_keys(PASSWORD)
        login_button.click()

        try:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_task_qdg41_28")))
        except TimeoutException:
            st.warning("Element not found. Waiting 10 seconds and retrying...")
            time.sleep(10)  # Wait 10 seconds

            try:
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_task_qdg41_28")))
            except TimeoutException:
                st.error("Element still not found after retrying.")
                st.code(driver.page_source[:1000])  # optional: show page source for debugging
                st.stop()
                time.sleep(10)

        all_tasks = driver.find_elements(By.CLASS_NAME, "_task_qdg41_28")
        boost_tasks = [t for t in all_tasks if "_boostTask_" in t.get_attribute("class")]
        total = len(boost_tasks)

        for idx, task in enumerate(boost_tasks, 1):
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", task)
                task.click()
                time.sleep(1)

                # Extract niche info
                try:
                    nich_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_conditions_h506w_88")))
                    nich_size_element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "_nicheSize_h506w_96")))
                    nich_text = nich_element.text.strip()
                    nich_size_text = nich_size_element.text.strip()
                except:
                    nich_text = "Not Found"
                    nich_size_text = "Not Found"

                # Go to Boost/Train metrics
                active_button = driver.find_element(By.XPATH, '/html/body/div/div/div/main/div/div[2]/div[2]/div[2]/div/div[1]/span[2]')
                driver.execute_script("arguments[0].click();", active_button)
                time.sleep(1)

                try:
                    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "_texts_zd90y_73")))
                    texts = driver.find_elements(By.CLASS_NAME, "_texts_zd90y_73")
                    boost = float(texts[0].text.strip().split('%')[0])
                    training = float(texts[1].text.strip().split('%')[0])
                except:
                    boost = training = "Not Found"

                clean_niche = nich_text.replace("\n", " ") if nich_text != "Not Found" else "Not Found"
                niche_size_match = re.search(r"(\d+)", nich_size_text)
                penetration_match = re.search(r"\(([^)]+)\)", nich_size_text)
                niche_size = niche_size_match.group(1) if niche_size_match else "Not Found"
                penetration = penetration_match.group(1) if penetration_match else "Not Found"

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
                st.error(f"{ordinal(idx)} iteration error: {e}")
                st.code(traceback.format_exc(), language="python")
                continue

        st.session_state.df_scraped = df_live

    except Exception as e:
        st.error(f"Main error: {e}")
        st.code(traceback.format_exc(), language="python")
        return df_live

    finally:
        driver.quit()
