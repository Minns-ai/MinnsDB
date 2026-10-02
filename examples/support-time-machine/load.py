"""Load endoflife.date support windows into MinnsDB as temporal graph edges.

(Product)-[supports]->(Release)        valid releaseDate .. eol
(Product)-[active_support]->(Release)  valid releaseDate .. support (end of bug fixes)
(Product)-[lts]->(Release)             valid lts date .. eol
"""
import glob, json, os, sys, urllib.request
from datetime import datetime, timezone

BASE = os.environ.get("MINNS_URL", "http://localhost:3000")
KEY = os.environ.get("MINNS_KEY")  # only needed with MINNS_AUTH_ENABLED=true
NAMES = {
    "alpine-linux": "Alpine", "android": "Android", "angular": "Angular", "centos": "CentOS",
    "debian": "Debian", "django": "Django", "dotnet": ".NET", "eclipse-temurin": "Temurin",
    "elasticsearch": "Elasticsearch", "electron": "Electron", "fedora": "Fedora", "go": "Go",
    "ios": "iOS", "kotlin": "Kotlin", "kubernetes": "Kubernetes", "laravel": "Laravel",
    "macos": "macOS", "mariadb": "MariaDB", "mongodb": "MongoDB", "mysql": "MySQL",
    "nextjs": "Next.js", "nginx": "nginx", "nodejs": "Node.js", "oracle-jdk": "Oracle JDK",
    "php": "PHP", "postgresql": "PostgreSQL", "python": "Python", "rails": "Rails",
    "redis": "Redis", "rhel": "RHEL", "ruby": "Ruby", "spring-boot": "Spring Boot",
    "terraform": "Terraform", "ubuntu": "Ubuntu", "vue": "Vue", "windows": "Windows",
}
EPOCH_FLOOR = "1970-01-01"  # valid_from is u64 nanoseconds, so nothing before 1970


def ns(d):
    return int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) * 10**9


def is_date(v):
    return isinstance(v, str) and len(v) == 10 and v[4] == "-"


def build():
    nodes, edges = [], []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eol", "*.json"))):
        slug = os.path.basename(f)[:-5]
        if slug not in NAMES:
            continue
        prod = NAMES[slug]
        nodes.append({"name": prod, "type": "concept", "properties": {"concept_type": "product", "slug": slug}})
        for c in json.load(open(f)):
            rel, eol = c.get("releaseDate"), c.get("eol")
            # eol: a date, False (still supported, open-ended edge) or True (ended, date unknown: skip)
            if not is_date(rel) or rel < EPOCH_FLOOR or not (is_date(eol) or eol is False):
                continue
            end = eol if is_date(eol) else None
            name = f"{prod} {c['cycle']}"
            props = {"concept_type": "release", "product": prod, "cycle": str(c["cycle"]),
                     "released": rel, "eol": end or "none announced"}
            if c.get("latest"):
                props["latest"] = str(c["latest"])
            nodes.append({"name": name, "type": "concept", "properties": props})

            def edge(label, start, stop):
                e = {"source": prod, "target": name, "label": label, "valid_from": ns(start),
                     "properties": {"until": stop or "open"}}
                if stop:
                    e["valid_until"] = ns(stop)
                edges.append(e)

            edge("supports", rel, end)
            sup = c.get("support")
            if is_date(sup) and (end is None or sup <= end):
                edge("active_support", rel, sup)
            elif sup is True:
                edge("active_support", rel, end)
            lts = c.get("lts")
            if is_date(lts) and (end is None or lts < end):
                edge("lts", lts, end)
            elif lts is True:
                edge("lts", rel, end)
    return nodes, edges


def main():
    nodes, edges = build()
    print(f"{len(nodes)} nodes, {len(edges)} edges", file=sys.stderr)
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = f"Bearer {KEY}"
    req = urllib.request.Request(BASE + "/api/graph/import", method="POST",
                                 data=json.dumps({"nodes": nodes, "edges": edges}).encode(), headers=headers)
    print(urllib.request.urlopen(req, timeout=120).read().decode()[:2000])


if __name__ == "__main__":
    main()
