import pandas as pd
import time
import re
import streamlit as st
import asyncio
from playwright.sync_api import sync_playwright

EMAIL = "inspector@fairgen.ai"
PASSWORD = "Inspector123!"

def scrap_boostresults(project_url):
    df_live = pd.DataFrame(columns=[
        "Niche", "Niche Size", "Penetration (%)", "Boost MAE (%)", "Training MAE (%)"
    ])

    progress_bar = st.empty()
    status_text = st.empty()
    df_placeholder = st.empty()

    def ordinal(n):
        return f"{n}st" if n == 1 else f"{n}nd" if n == 2 else f"{n}rd" if n == 3 else f"{n}th"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            page.goto(project_url)

            page.fill('input[name="email"]', EMAIL)
            page.fill('input[name="password"]', PASSWORD)
            page.click('button[type="submit"]')
            page.wait_for_selector('._task_qdg41_28')
            time.sleep(2)

            tasks = page.query_selector_all('._task_qdg41_28')
            boost_tasks = [t for t in tasks if "_boostTask_" in t.get_attribute("class")]
            total = len(boost_tasks)

            for idx, task in enumerate(boost_tasks, 1):
                try:
                    task.scroll_into_view_if_needed()
                    task.click()
                    time.sleep(1)

                    try:
                        nich_text = page.query_selector('._conditions_h506w_88').inner_text().strip()
                        nich_size_text = page.query_selector('._nicheSize_h506w_96').inner_text().strip()
                    except:
                        nich_text = "Not Found"
                        nich_size_text = "Not Found"

                    page.click('xpath=//*[@id="app"]/div/main/div/div[2]/div[2]/div[2]/div/div[1]/span[2]')
                    time.sleep(1)

                    texts = page.query_selector_all('._texts_zd90y_73')
                    if len(texts) >= 2:
                        boost = float(texts[0].inner_text().strip().replace('%', ''))
                        training = float(texts[1].inner_text().strip().replace('%', ''))
                    else:
                        boost = training = "Not Found"

                    clean_niche = nich_text.replace("\n", " ")
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
                    status_text.text(f"Processed {idx} of {total} boosts...")
                    df_placeholder.dataframe(df_live)

                except Exception as e:
                    st.error(f"{ordinal(idx)} iteration error: {e}")
                    continue

            browser.close()
        return df_live

    except Exception as e:
        st.error(f"Main error: {e}")
        return df_live
