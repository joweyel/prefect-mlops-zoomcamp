import httpx
from prefect import flow, task

@task(retries=4, retry_delay_seconds=1.0, log_prints=True)
def fetch_cat_fact():
    cat_fact = httpx.get("https://f3-vyx5c2hfpq-ue.a.run.app/")
    if cat_fact.status_code >= 400:
        raise Exception()
    print(cat_fact.text)
    return cat_fact.text

@flow
def fetch():
    fact = fetch_cat_fact()
    return fact

if __name__ == "__main__":
    result = fetch()