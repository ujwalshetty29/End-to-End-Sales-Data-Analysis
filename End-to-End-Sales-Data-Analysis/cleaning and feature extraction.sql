USE project;


DELETE t1
FROM orde t1
JOIN (
    SELECT `Order Id`, `Product Id`, MIN(id) AS min_id
    FROM orde
    GROUP BY `Order Id`, `Product Id`
) t2
ON t1.`Order Id` = t2.`Order Id` 
AND t1.`Product Id` = t2.`Product Id`
AND t1.id <> t2.min_id;


UPDATE orde
SET City = COALESCE(City, 'Unknown'),
    State = COALESCE(State, 'Unknown'),
    Region = COALESCE(Region, 'Unknown'),
    `Postal Code` = COALESCE(`Postal Code`, '000000'),
    Segment = COALESCE(Segment, 'Unknown'),
    Category = COALESCE(Category, 'Unknown'),
    `Sub Category` = COALESCE(`Sub Category`, 'Unknown'),
    `Order Date` = COALESCE(`Order Date`, CURDATE());


UPDATE orde
SET Quantity = CASE WHEN Quantity < 0 THEN 0 ELSE Quantity END,
    `cost price` = CASE WHEN `cost price` < 0 THEN 0 ELSE `cost price` END,
    `List Price` = CASE WHEN `List Price` < 0 THEN 0 ELSE `List Price` END;


UPDATE orde
SET `Discount Percent` = CASE 
                            WHEN `Discount Percent` < 0 THEN 0
                            WHEN `Discount Percent` > 100 THEN 100
                            ELSE `Discount Percent`
                         END;


SELECT
    *,
    STR_TO_DATE(`Order Date`, '%Y-%m-%d') AS Order_Date,   
    (`List Price` * `Discount Percent` / 100.0) AS Discount_Value,
    (`List Price` - (`List Price` * `Discount Percent` / 100.0)) AS Sales_Price,
    ( (`List Price` - (`List Price` * `Discount Percent` / 100.0)) * Quantity
      - (`cost price` * Quantity) ) AS Profit

FROM orde;
