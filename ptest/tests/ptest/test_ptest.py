import dotenv
import pytest
import ptest


@pytest.fixture(scope="session", autouse=True)
def load_env():
    dotenv.load_dotenv()


@pytest.mark.asyncio
def test_happy_path():
    assert ptest.__version__ == "0.0.0-dev"
