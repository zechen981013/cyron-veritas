from fastapi.testclient import TestClient

from cyron_veritas.web.server import app

c = TestClient(app)


def test_health_and_portfolios():
    assert c.get("/api/health").json() == {"ok": True}
    names = [p["name"] for p in c.get("/api/portfolios").json()]
    assert "sample_portfolio.csv" in names


def test_bad_inputs_rejected():
    assert c.get("/api/jobs/nope").status_code == 404
    assert c.post("/api/research/start", json={"ticker": "12-34!"}).status_code == 400
    assert c.post("/api/risk/run", json={"portfolio": "../pyproject.toml"}).status_code == 400


def test_market_detection():
    from cyron_veritas.data.financials import market_of

    assert market_of("000333") == ("A", "000333")
    assert market_of("700") == ("HK", "00700")
    assert market_of("aapl") == ("US", "AAPL")
