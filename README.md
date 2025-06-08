# Bank Statement Analyzer

A powerful web-based tool for analyzing bank statements in PDF format. This application helps you understand your spending patterns, track transactions, and generate detailed financial insights.

## Features

### 1. PDF Statement Processing
- Automatically extracts transaction data from PDF bank statements
- Parses dates, times, descriptions, amounts, and transaction details
- Supports multiple statement formats

### 2. Transaction Analysis
- Separates deposits and withdrawals
- Calculates running balances
- Provides transaction counts and totals
- Shows detailed channel information and transaction specifics
- AI-powered transaction categorization

### 3. Interactive Data Management
- Remove/restore individual transactions for "what-if" analysis
- Real-time updates of totals and statistics
- Track changes with a detailed changes summary
- View the impact of excluded transactions

### 4. Visual Analytics
- Interactive pie chart showing spending by category
- Color-coded transaction amounts (green for deposits, red for withdrawals)
- Clear presentation of transaction channels and details

### 5. Export Capabilities
- Export to Excel with formatted sheets
- Includes summary statistics
- Color-coded cells and proper number formatting
- Transaction categorization and analysis

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd bank-statement-analyzer
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Analyzing a Statement

1. **Upload Your Statement**
   - Click the upload button on the main page
   - Select your bank statement PDF file
   - Click "Analyze Statement"

2. **View Transaction Details**
   - All transactions will be displayed in a detailed table
   - View date, time, description, amounts, and channel information
   - Check the running balance for each transaction

3. **Manage Transactions**
   - Use the "Remove" button to temporarily exclude transactions
   - Click "Undo" to restore removed transactions
   - Watch the summary statistics update in real-time
   - View the impact of your changes in the Changes Summary section

4. **Analyze Spending Patterns**
   - Check the pie chart for spending category breakdown
   - View total deposits and withdrawals
   - Monitor transaction counts and net balance
   - Review AI-generated insights about your spending patterns

5. **Export Data**
   - Click "Export to Excel" to download a detailed report
   - Get formatted sheets with all transaction details
   - Access summary statistics and categorized spending
   - Use for record-keeping or further analysis

## Privacy & Security

- All processing is done locally on your machine
- No data is sent to external servers (except for AI analysis if enabled)
- PDF files are processed in memory and not stored permanently
- No sensitive banking information is retained after analysis

## Requirements

- Python 3.7 or higher
- Flask web framework
- pdfplumber for PDF processing
- pandas and openpyxl for Excel export
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 