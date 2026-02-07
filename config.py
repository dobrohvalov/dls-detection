from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_URL: str
    CURRENT_USERNAME: str = None
    CURRENT_PASSWORD: str = None
    DEBUG_MODE: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'


settings = Settings()
