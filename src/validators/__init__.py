from concurrent.futures import ThreadPoolExecutor

from src.config import get_secret
from .spf            import check_spf
from .dkim           import check_dkim
from .dmarc          import check_dmarc
from .ip_reputation  import check_ip_reputation, check_domain_reputation
from .geolocation    import geolocate_ip
from .file_reputation import check_file_hash, check_url

class EmailSecurityValidator:
    def __init__(self):
        pass

    # 🟢 Tutti i metodi sotto sono ora rientrati di 4 spazi: appartengono CORRETTAMENTE alla classe!
    def check_spf(self, sender_ip: str, mail_from: str, helo_domain: str = "") -> dict:
        return check_spf(None, sender_ip, mail_from, helo_domain)

    def check_dkim(self, raw_eml_bytes: bytes) -> dict:
        return check_dkim(raw_eml_bytes)

    def check_dmarc(self, from_address: str, spf_result: str, spf_domain: str, dkim_results: list) -> dict:
        return check_dmarc(None, from_address, spf_result, spf_domain, dkim_results)
    
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

    # 🟢 The pipeline now has its 4-space indentation and is a full class method
    def pipeline_analisi_veloce(self, sender_ip: str, mail_from: str, domain: str, sha256: str = "") -> dict:
        """
        Esegue i controlli di reputazione e geolocalizzazione in PARALLELO.
        Evita che l'applicazione si blocchi attendendo le risposte sequenziali delle API.
        """
        risultati = {}
        
        # Sfruttiamo un pool di thread per le chiamate HTTP concorrenti
        with ThreadPoolExecutor(max_workers=4) as executor:
            futuro_ip = executor.submit(self.check_ip_reputation, sender_ip)
            futuro_dom = executor.submit(self.check_domain_reputation, domain)
            futuro_geo = executor.submit(self.geolocate_ip, sender_ip)
            futuro_file = executor.submit(self.check_file_hash, sha256) if sha256 else None

            risultati["ip_reputation"] = futuro_ip.result()
            risultati["domain_reputation"] = futuro_dom.result()
            risultati["geolocation"] = futuro_geo.result()
            risultati["file_reputation"] = futuro_file.result() if futuro_file else {"status": "skipped", "message": "Nessun file"}

        return risultati
