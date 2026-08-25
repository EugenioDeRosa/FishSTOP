from src.config import get_secret
from .ip_reputation  import check_ip_reputation, check_domain_reputation
from .geolocation    import geolocate_ip
from .file_reputation import check_file_hash, check_url

class EmailSecurityValidator:
    def check_domain_reputation(self, domain: str, api_key: str | None = None) -> dict:
        return check_domain_reputation(domain, api_key=api_key)

    def geolocate_ip(self, ip: str) -> dict:
        return geolocate_ip(ip)

    def check_file_hash(self, sha256: str, api_key: str | None = None) -> dict:
        return check_file_hash(
            api_key if api_key is not None else get_secret("VIRUSTOTAL_API_KEY"),
            sha256,
        )

    def check_url_reputation(self, url: str, api_key: str | None = None) -> dict:
        return check_url(
            api_key if api_key is not None else get_secret("VIRUSTOTAL_API_KEY"),
            url,
        )
    
    def check_ip_reputation(self, ip: str, api_key: str | None = None) -> dict:
        return check_ip_reputation(ip, api_key=api_key)
