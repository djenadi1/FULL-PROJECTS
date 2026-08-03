/* ==========================================================
                SUPERSTORE DATA CLEANING
   ========================================================== */


/* -------------------------
   Inspect the raw dataset
   ------------------------- */

SELECT *
FROM superstore;


/* -------------------------
   Create a staging table

   Always perform cleaning on a copy of the data
   to preserve the original dataset.
   ------------------------- */

CREATE TABLE superstore_staging_0
LIKE superstore;

INSERT INTO superstore_staging_0
SELECT *
FROM superstore;


/* -------------------------
   Duplicate check

   No duplicate rows were found.
   ------------------------- */


/* -------------------------
   Standardizing data

   Remove leading and trailing spaces from all
   text columns using TRIM().
   ------------------------- */

UPDATE superstore_staging_0
SET
    `Order ID` = TRIM(`Order ID`),
    `Ship Mode` = TRIM(`Ship Mode`),
    `Customer ID` = TRIM(`Customer ID`),
    `Customer Name` = TRIM(`Customer Name`),
    `Segment` = TRIM(`Segment`),
    `Country` = TRIM(`Country`),
    `City` = TRIM(`City`),
    `State` = TRIM(`State`),
    `Region` = TRIM(`Region`),
    `Product ID` = TRIM(`Product ID`),
    `Category` = TRIM(`Category`),
    `Sub-Category` = TRIM(`Sub-Category`),
    `Product Name` = TRIM(`Product Name`);


/* -------------------------
   Looking for inconsistent
   values within categorical
   columns.
   ------------------------- */

SELECT DISTINCT `Ship Mode`
FROM superstore_staging_0
ORDER BY `Ship Mode`;

SELECT DISTINCT `Segment`
FROM superstore_staging_0
ORDER BY `Segment`;

SELECT DISTINCT `Country`
FROM superstore_staging_0
ORDER BY `Country`;

SELECT DISTINCT `State`
FROM superstore_staging_0
ORDER BY `State`;

SELECT DISTINCT `Region`
FROM superstore_staging_0
ORDER BY `Region`;

SELECT DISTINCT `Category`
FROM superstore_staging_0
ORDER BY `Category`;

SELECT DISTINCT `Sub-Category`
FROM superstore_staging_0
ORDER BY `Sub-Category`;


/* -------------------------
   Inspect numeric columns
   for unusual minimum and
   maximum values.
   ------------------------- */

SELECT
    MIN(Sales),
    MAX(Sales),
    MIN(Quantity),
    MAX(Quantity),
    MIN(Discount),
    MAX(Discount),
    MIN(Profit),
    MAX(Profit)
FROM superstore_staging_0;


/* -------------------------
   Looking for NULL or empty
   values in every column.
   ------------------------- */

SELECT
    SUM(`Row ID` IS NULL OR `Row ID` = '') AS RowID_Missing,
    SUM(`Order ID` IS NULL OR `Order ID` = '') AS OrderID_Missing,
    SUM(`Order Date` IS NULL OR `Order Date` = '') AS OrderDate_Missing,
    SUM(`Ship Date` IS NULL OR `Ship Date` = '') AS ShipDate_Missing,
    SUM(`Ship Mode` IS NULL OR `Ship Mode` = '') AS ShipMode_Missing,
    SUM(`Customer ID` IS NULL OR `Customer ID` = '') AS CustomerID_Missing,
    SUM(`Customer Name` IS NULL OR `Customer Name` = '') AS CustomerName_Missing,
    SUM(`Segment` IS NULL OR `Segment` = '') AS Segment_Missing,
    SUM(`Country` IS NULL OR `Country` = '') AS Country_Missing,
    SUM(`City` IS NULL OR `City` = '') AS City_Missing,
    SUM(`State` IS NULL OR `State` = '') AS State_Missing,
    SUM(`Postal Code` IS NULL OR `Postal Code` = '') AS PostalCode_Missing,
    SUM(`Region` IS NULL OR `Region` = '') AS Region_Missing,
    SUM(`Product ID` IS NULL OR `Product ID` = '') AS ProductID_Missing,
    SUM(`Category` IS NULL OR `Category` = '') AS Category_Missing,
    SUM(`Sub-Category` IS NULL OR `Sub-Category` = '') AS SubCategory_Missing,
    SUM(`Product Name` IS NULL OR `Product Name` = '') AS ProductName_Missing,
    SUM(Sales IS NULL OR Sales = '') AS Sales_Missing,
    SUM(Quantity IS NULL OR Quantity = '') AS Quantity_Missing,
    SUM(Discount IS NULL OR Discount = '') AS Discount_Missing,
    SUM(Profit IS NULL OR Profit = '') AS Profit_Missing;


/* -------------------------
   Convert date columns from
   text (MM/DD/YYYY) into
   MySQL DATE format.
   ------------------------- */

UPDATE superstore_staging_0
SET
    `Order Date` = STR_TO_DATE(`Order Date`, '%m/%d/%Y'),
    `Ship Date` = STR_TO_DATE(`Ship Date`, '%m/%d/%Y');


/* -------------------------
   Change both columns to
   the DATE data type.
   ------------------------- */

ALTER TABLE superstore_staging_0
MODIFY COLUMN `Order Date` DATE,
MODIFY COLUMN `Ship Date` DATE;


/* -------------------------
   Final inspection of the
   cleaned dataset.
   ------------------------- */

SELECT *
FROM superstore_staging_0;