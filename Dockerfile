FROM python:3.9-slim

WORKDIR /app

# Enable required sources and install OpenJDK 11 JRE
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    software-properties-common \
    build-essential \
    git

# Add Debian backports to get openjdk-11
RUN echo "deb http://deb.debian.org/debian bullseye main" >> /etc/apt/sources.list \
 && echo "deb http://security.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list \
 && echo "deb http://deb.debian.org/debian bullseye-updates main" >> /etc/apt/sources.list \
 && apt-get update && apt-get install -y openjdk-11-jre-headless \
 && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

COPY requirements.txt ./
COPY src/ ./src/
COPY StackedEnsemble_BestOfFamily_1_AutoML_3_20250807_141418.model ./

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
