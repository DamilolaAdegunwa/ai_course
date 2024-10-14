import ssl
import certifi
from transformers import pipeline
from requests.adapters import HTTPAdapter
# from requests.packages.urllib3.poolmanager import PoolManager
from requests import Session

# Define an adapter to use certifi for SSL certificates
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context(cafile=certifi.where())
        # kwargs['ssl_context'] = context
        kwargs['ssl_context'] = None
        return super(SSLAdapter, self).init_poolmanager(*args, **kwargs)

# Create a session and mount the SSL adapter
session = Session()
adapter = SSLAdapter()
session.mount('https://', adapter)

# Pass the session into the pipeline
# classifier = pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", use_auth_token=None, session=session)
classifier = pipeline(task="text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", session=session)

# Example usage
print(classifier(["You are the best", "Get Lost"]))
