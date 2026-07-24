class DataSource:
    def __init__(self, url, name, source_license, description):
        self.url = url,
        self.name = name,
        self.licence = source_license,
        self.desc = description

phishing_email_dataset = DataSource(url="https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?select=CEAS_08.csv", name="Phishing Email Dataset", source_license="CC BY-SA 4.0", description="This dataset contains approx 82,500 emails; 42,891 spam and 39,595 legitimate emails")

phishing_url_dataset = DataSource(url="https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector", name="Phishing website Detector", source_license="Unknown", description="This dataset contains 11,055 records with 31 features related to phishing websites.")

print(f"Email: {phishing_email_dataset} \nURL: {phishing_url_dataset}")