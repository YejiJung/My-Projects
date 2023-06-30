-- Car Dealership Sales Datbase

-- 1. How many counts of vehicle sold in different location in different years?

SELECT 
	year(Paymentdate),	Locationname,
	COUNT(*)
From Final_Project
Group by 1,2


-- 2. How many counts of vehicle sold in different warehouse in different years

Select 
	year(paymentdate), Warehousename,
	COUNT(*)
From Final_Project
Group by 1,2


-- 3. How many counts of vehicle category sold in different years

SELECT 	year(Paymentdate),
	Vehiclecategory,
	COUNT(*)
From Final_Project
Group by 1,2

-- 4. How much is the total amount of each vehicle in different years?

SELECT 	year(Paymentdate), 
	Vehiclename, 
	Sum(`Orderprice`)
FROM `Final_Project` 
Group by 1, 2

-- 5. Which Vehicle was the highest sale in 2015

SELECT 	year(Paymentdate),
	Vehiclename,
	Count(*)
FROM `Final_Project` 
Where paymentdate = 2015
Group by 1,2

-- 6. How many customers use different payment type in different years

SELECT 	year(Paymentdate),
	Paymenttype,
	Count(*)
FROM `Final_Project` 
Group by 1,2


-- 7. Display sales order amount per year 

SELECT Year(Orderdate),Count(Orderdate)
FROM `Final_Project` 
Group by 1


-- 8. Display Customer name, Vehicle category that was sold in more than 20 times

SELECT	Year(Paymentdate),
	Customername,
	Vehiclecategory,
	COUNT(1) soldtimes
FROM Final_Project
GROUP BY 1, 2, 3
HAVING COUNT(1)> 20

-- 


