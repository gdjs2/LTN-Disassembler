# docker run -it --name ltn-disassembler   -v `pwd`/LTN-Disassembler:/app   ltn-disassembler
# Use official Python 3.11 image
FROM python:3.11-slim

# Install system dependencies for Python modules, gcc, and graphviz
RUN apt-get update && apt-get install -y wget unzip \
    gcc \
    graphviz \
    libgraphviz-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    libbz2-dev \
    libffi-dev \
    libssl-dev \
    libncurses5-dev \
    libreadline-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and extract OpenJDK 21
RUN wget -O /tmp/jdk-21.0.7_linux-x64_bin.tar.gz https://download.oracle.com/java/21/archive/jdk-21.0.7_linux-x64_bin.tar.gz \
    && mkdir -p /opt/jdk-21 \
    && tar -xzf /tmp/jdk-21.0.7_linux-x64_bin.tar.gz -C /opt/jdk-21 --strip-components=1 \
    && rm /tmp/jdk-21.0.7_linux-x64_bin.tar.gz

ENV JAVA_HOME=/opt/jdk-21
ENV PATH="$JAVA_HOME/bin:$PATH"

# Download and extract Ghidra 11.3.2
RUN wget -O /tmp/ghidra_11.3.2_PUBLIC_20250415.zip https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.3.2_build/ghidra_11.3.2_PUBLIC_20250415.zip \
    && unzip /tmp/ghidra_11.3.2_PUBLIC_20250415.zip -d /opt \
    && rm /tmp/ghidra_11.3.2_PUBLIC_20250415.zip

ENV GHIDRA_INSTALL_DIR=/opt/ghidra_11.3.2_PUBLIC

# Set working directory
WORKDIR /app

# Copy all files from LTN-Disassembler
COPY . /app

# Install Python dependencies and any system dependencies for them
RUN pip install --no-cache-dir -r requirements.txt

# Default command (can be changed as needed)
CMD ["bash"]
