# docker run -it --name ltn-disassembler --gpus all  -v `pwd`/LTN-Disassembler:/app   ltn-disassembler
# Use official Python 3.11 image
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install system dependencies for Python modules, gcc, and graphviz
RUN apt-get update && apt-get install -y wget unzip \
    gcc \
    python3.11 \
    python3-pip \
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
    sudo \
    && rm -rf /var/lib/apt/lists/*
    
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a regular user to avoid permission issues
RUN useradd -m -s /bin/bash -u 1000 ltn && \
    mkdir -p /app && \
    chown -R ltn:ltn /app && \
    # Give the user sudo access (optional, remove if not needed)
    echo "ltn ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /app

# Copy all files from LTN-Disassembler
COPY --chown=ltn:ltn . /app
# Install Python dependencies and any system dependencies for them
RUN pip install --no-cache-dir -r requirements.txt

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
    && rm /tmp/ghidra_11.3.2_PUBLIC_20250415.zip \
    && chown -R ltn:ltn /opt/ghidra_11.3.2_PUBLIC

ENV GHIDRA_INSTALL_DIR=/opt/ghidra_11.3.2_PUBLIC

# Switch to regular user
USER ltn

# Default command (can be changed as needed)
CMD ["bash"]
