# NEURAX Demo Applications Makefile

# Compiler settings
CC ?= gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -L../software/lib/lib -static -lneurax -lm

# Debug build
ifdef DEBUG
CFLAGS += -g -DNEURAX_DEBUG -O0
endif

# Directories
SRC_DIR = simple_cnn
BLUR_SRC_DIR = image_blur
VIDEO_SRC_DIR = realtime_video
BUILD_DIR = build
BIN_DIR = bin

# Include paths
INCLUDES = -I../software/lib/include

# Source files
CNN_SOURCES = $(SRC_DIR)/simple_cnn_demo.c
BLUR_SOURCES = $(BLUR_SRC_DIR)/image_blur_demo.c
VIDEO_SOURCES = $(VIDEO_SRC_DIR)/realtime_video_demo.c

# Output binaries
CNN_TARGET = $(BIN_DIR)/simple_cnn_demo
BLUR_TARGET = $(BIN_DIR)/image_blur_demo
VIDEO_TARGET = $(BIN_DIR)/realtime_video_demo

# Default target
.PHONY: all clean install blur-demo video-demo

all: $(CNN_TARGET) $(BLUR_TARGET) $(VIDEO_TARGET)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Build demo application
$(CNN_TARGET): $(CNN_SOURCES) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(CNN_SOURCES) -o $@ $(LDFLAGS)
	@echo "CNN demo application built: $@"

# Build image blur demo
$(BLUR_TARGET): $(BLUR_SOURCES) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(BLUR_SOURCES) -o $@ $(LDFLAGS)
	@echo "Image blur demo application built: $@"

# Build real-time video demo  
$(VIDEO_TARGET): $(VIDEO_SOURCES) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) $(VIDEO_SOURCES) -o $@ $(LDFLAGS) -lpthread -ljpeg -lv4l2
	@echo "Real-time video demo application built: $@"

# Build specific demos
blur-demo: $(BLUR_TARGET)
cnn-demo: $(CNN_TARGET)
video-demo: $(VIDEO_TARGET)

# Development build
dev:
	$(MAKE) CC=gcc DEBUG=1 all

# Install demo applications
install: all
	@echo "Installing demo applications..."
	sudo cp $(CNN_TARGET) /usr/local/bin/
	sudo cp $(BLUR_TARGET) /usr/local/bin/
	sudo cp $(VIDEO_TARGET) /usr/local/bin/
	@echo "Installation complete"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(BIN_DIR)
	@echo "Clean complete"

# Run demo
run: $(CNN_TARGET)
	@echo "Running simple CNN demo..."
	LD_LIBRARY_PATH=../../software/lib/lib:$$LD_LIBRARY_PATH $(CNN_TARGET)

# Run benchmark
benchmark: $(CNN_TARGET)
	@echo "Running performance benchmark..."
	LD_LIBRARY_PATH=../../software/lib/lib:$$LD_LIBRARY_PATH $(CNN_TARGET) --benchmark

# Run image blur demo
run-blur: $(BLUR_TARGET)
	@echo "Running image blur demo..."
	@echo "Usage: LD_LIBRARY_PATH=../../software/lib/lib:$$LD_LIBRARY_PATH $(BLUR_TARGET) <input.bmp> <output.bmp> [options]"
	@echo "Use --create-sample to generate a test image first"

# Run real-time video demo
run-video: $(VIDEO_TARGET)
	@echo "Running real-time video demo..."
	@echo "Open http://localhost:8080/stream in your browser"
	LD_LIBRARY_PATH=../../software/lib/lib:$$LD_LIBRARY_PATH $(VIDEO_TARGET)

# Create sample image and run blur
demo-blur: $(BLUR_TARGET)
	@echo "Creating sample image and running blur demo..."
	@echo "Using convenience script for easier testing..."
	./run_blur_demo.sh sample_blur.bmp --kernel-size 7 --sigma 2.0

# Run blur demo with test image
test-blur: $(BLUR_TARGET)
	@echo "Running blur demo with test image..."
	./run_blur_demo.sh

# Show help
help:
	@echo "NEURAX Demo Makefile"
	@echo "===================="
	@echo "Available targets:"
	@echo "  all       - Build all demo applications"
	@echo "  cnn-demo  - Build CNN demo only"
	@echo "  blur-demo - Build image blur demo only"
	@echo "  video-demo - Build real-time video demo only"
	@echo "  dev       - Build for development"
	@echo "  install   - Install demos to system"
	@echo "  clean     - Remove build artifacts"
	@echo "  run       - Run simple CNN demo"
	@echo "  benchmark - Run performance benchmark"
	@echo "  run-blur  - Show usage for image blur demo"
	@echo "  run-video - Run real-time video demo"
	@echo "  demo-blur - Run blur demo with test image and custom parameters"
	@echo "  test-blur - Run blur demo with default parameters"
	@echo "  help      - Show this help"
