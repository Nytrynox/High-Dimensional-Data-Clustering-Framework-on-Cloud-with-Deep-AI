#!/bin/bash

# Database-Free Setup Script
# Sets up the clustering framework without any database dependencies

set -e

echo "🆓 Setting up Database-Free Clustering Framework..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 found${NC}"

# Create virtual environment
echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
python3 -m venv clustering-env

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source clustering-env/bin/activate

# Install minimal requirements
echo -e "${YELLOW}📚 Installing minimal requirements...${NC}"
pip install --upgrade pip
pip install -r requirements_minimal.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data results models cache logs temp experiments

echo -e "${GREEN}✅ Directories created${NC}"

# Generate sample data
echo -e "${YELLOW}🎲 Generating sample data...${NC}"
python main_simple.py generate-sample

echo -e "${GREEN}✅ Sample data generated${NC}"

# Test basic functionality
echo -e "${YELLOW}🧪 Testing basic clustering...${NC}"
python main_simple.py quick data/sample_data.csv --algorithm kmeans --n-clusters 3

echo -e "${GREEN}✅ Basic clustering test passed${NC}"

# Show status
echo -e "${YELLOW}📊 System status:${NC}"
python main_simple.py status

echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Activate environment: source clustering-env/bin/activate"
echo "2. Run clustering: python main_simple.py cluster data/sample_data.csv"
echo "3. View experiments: python main_simple.py experiments"
echo "4. Optional API: pip install fastapi uvicorn[standard] && python main_simple.py serve"
echo ""
echo -e "${GREEN}💡 Everything runs locally - no databases, no cloud dependencies!${NC}"
