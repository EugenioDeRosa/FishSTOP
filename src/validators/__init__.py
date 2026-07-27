from src.config import get_secret
from .ip_reputation  import check_ip_reputation, check_domain_reputation
from .geolocation    import geolocate_ip
from .file_reputation import check_file_hash, check_url

class EmailSecurityValidator:
    def check_domain_reputation(self, domain: str) -> dict:
        return check_domain_reputation(domain)  

    def geolocate_ip(self, ip: str) -> dict:
        return geolocate_ip(ip)

    def check_file_hash(self, sha256: str) -> dict:
        return check_file_hash(get_secret("VIRUSTOTAL_API_KEY"), sha256)

    def check_url_reputation(self, url: str) -> dict:
        return check_url(get_secret("VIRUSTOTAL_API_KEY"), url)
    
    def check_ip_reputation(self, ip: str) -> dict:
        return check_ip_reputation(ip)
