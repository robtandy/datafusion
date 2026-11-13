-- warmup
select count(*) from lineitem, orders where l_orderkey = o_orderkey and l_quantity > 10;

-- q2
select
            s_acctbal,
            s_name,
            n_name,
            p_partkey,
            p_mfgr,
            s_address,
            s_phone,
            s_comment
        from
            part,
            supplier,
            partsupp,
            nation,
            region
        where
            p_partkey = ps_partkey
            and s_suppkey = ps_suppkey
            and p_size = 15
            and p_type like '%BRASS'
            and s_nationkey = n_nationkey
            and n_regionkey = r_regionkey
            and r_name = 'EUROPE'
            and ps_supplycost = (
                select
                    min(ps_supplycost)
                from
                    partsupp,
                    supplier,
                    nation,
                    region
                where
                    p_partkey = ps_partkey
                    and s_suppkey = ps_suppkey
                    and s_nationkey = n_nationkey
                    and n_regionkey = r_regionkey
                    and r_name = 'EUROPE'
            )
        order by
            s_acctbal desc,
            n_name,
            s_name,
            p_partkey
        limit 100;

-- q2_duckdb_join_order
select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment from part join (partsupp join (supplier join (nation join region on n_regionkey = r_regionkey) on s_nationkey = n_nationkey) on ps_suppkey = s_suppkey) on p_partkey = ps_partkey where p_size = 15 and p_type like '%BRASS' and r_name = 'EUROPE' and ps_supplycost = ( select min(ps_supplycost) from partsupp join (supplier join (nation join region on n_regionkey = r_regionkey) on s_nationkey = n_nationkey) on ps_suppkey = s_suppkey where p_partkey = ps_partkey and r_name = 'EUROPE' ) order by s_acctbal desc, n_name, s_name, p_partkey limit 100;

-- q5
select
            n_name,
            sum(l_extendedprice * (1 - l_discount)) as revenue
        from
            customer,
            orders,
            lineitem,
            supplier,
            nation,
            region
        where
            c_custkey = o_custkey
            and l_orderkey = o_orderkey
            and l_suppkey = s_suppkey
            and c_nationkey = s_nationkey
            and s_nationkey = n_nationkey
            and n_regionkey = r_regionkey
            and r_name = 'ASIA'
            and o_orderdate >= date '1994-01-01'
            and o_orderdate < date '1994-01-01' + interval '1' year
        group by
            n_name
        order by
            revenue desc;

-- q5_duckdb_join_order
select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue from lineitem join (orders join (supplier join (customer join (nation join region on n_regionkey = r_regionkey) on c_nationkey = n_nationkey) on s_nationkey = c_nationkey) on o_custkey = c_custkey) on l_orderkey = o_orderkey and l_suppkey = s_suppkey where r_name = 'ASIA' and o_orderdate >= date '1994-01-01' and o_orderdate < date '1994-01-01' + interval '1' year group by n_name order by revenue desc;

-- q7
select
            supp_nation,
            cust_nation,
            l_year,
            sum(volume) as revenue
        from
            (
                select
                    n1.n_name as supp_nation,
                    n2.n_name as cust_nation,
                    extract(year from l_shipdate) as l_year,
                    l_extendedprice * (1 - l_discount) as volume
                from
                    supplier,
                    lineitem,
                    orders,
                    customer,
                    nation n1,
                    nation n2
                where
                    s_suppkey = l_suppkey
                    and o_orderkey = l_orderkey
                    and c_custkey = o_custkey
                    and s_nationkey = n1.n_nationkey
                    and c_nationkey = n2.n_nationkey
                    and (
                        (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY')
                        or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE')
                    )
                    and l_shipdate between date '1995-01-01' and date '1996-12-31'
            ) as shipping
        group by
            supp_nation,
            cust_nation,
            l_year
        order by
            supp_nation,
            cust_nation,
            l_year;

-- q7_duckdb_join_order
select supp_nation, cust_nation, l_year, sum(volume) as revenue from ( select n1.n_name as supp_nation, n2.n_name as cust_nation, extract(year from l_shipdate) as l_year, l_extendedprice * (1 - l_discount) as volume from (customer join nation n2 on c_nationkey = n2.n_nationkey) join (orders join (lineitem join (supplier join nation n1 on s_nationkey = n1.n_nationkey) on l_suppkey = s_suppkey) on o_orderkey = l_orderkey) on c_custkey = o_custkey and ( (n1.n_name = 'FRANCE' and n2.n_name = 'GERMANY') or (n1.n_name = 'GERMANY' and n2.n_name = 'FRANCE') ) where l_shipdate between date '1995-01-01' and date '1996-12-31' ) as shipping group by supp_nation, cust_nation, l_year order by supp_nation, cust_nation, l_year;

-- q8
select
            o_year,
            sum(case
                when nation = 'BRAZIL' then volume
                else 0
            end) / sum(volume) as mkt_share
        from
            (
                select
                    extract(year from o_orderdate) as o_year,
                    l_extendedprice * (1 - l_discount) as volume,
                    n2.n_name as nation
                from
                    part,
                    supplier,
                    lineitem,
                    orders,
                    customer,
                    nation n1,
                    nation n2,
                    region
                where
                    p_partkey = l_partkey
                    and s_suppkey = l_suppkey
                    and l_orderkey = o_orderkey
                    and o_custkey = c_custkey
                    and c_nationkey = n1.n_nationkey
                    and n1.n_regionkey = r_regionkey
                    and r_name = 'AMERICA'
                    and s_nationkey = n2.n_nationkey
                    and o_orderdate between date '1995-01-01' and date '1996-12-31'
                    and p_type = 'ECONOMY ANODIZED STEEL'
            ) as all_nations
        group by
            o_year
        order by
            o_year;

-- q8_duckdb_join_order
select o_year, sum(case when nation = 'BRAZIL' then volume else 0 end) / sum(volume) as mkt_share from ( select extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) as volume, n2.n_name as nation from nation n2 join (supplier join (part join (lineitem join (orders join (customer join (nation n1 join region on n1.n_regionkey = r_regionkey) on c_nationkey = n1.n_nationkey) on o_custkey = c_custkey) on l_orderkey = o_orderkey) on p_partkey = l_partkey) on s_suppkey = l_suppkey) on s_nationkey = n2.n_nationkey where r_name = 'AMERICA' and o_orderdate between date '1995-01-01' and date '1996-12-31' and p_type = 'ECONOMY ANODIZED STEEL' ) as all_nations group by o_year order by o_year;

-- q9
select
            nation,
            o_year,
            sum(amount) as sum_profit
        from
            (
                select
                    n_name as nation,
                    extract(year from o_orderdate) as o_year,
                    l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
                from
                    part,
                    supplier,
                    lineitem,
                    partsupp,
                    orders,
                    nation
                where
                    s_suppkey = l_suppkey
                    and ps_suppkey = l_suppkey
                    and ps_partkey = l_partkey
                    and p_partkey = l_partkey
                    and o_orderkey = l_orderkey
                    and s_nationkey = n_nationkey
                    and p_name like '%green%'
            ) as profit
        group by
            nation,
            o_year
        order by
            nation,
            o_year desc;

-- q9_duckdb_join_order
select nation, o_year, sum(amount) as sum_profit from ( select n_name as nation, extract(year from o_orderdate) as o_year, l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount from lineitem join ((part join partsupp on ps_partkey = p_partkey) join (supplier join nation on s_nationkey = n_nationkey) on ps_suppkey = s_suppkey) on l_partkey = p_partkey and l_suppkey = s_suppkey join orders on o_orderkey = l_orderkey where p_name like '%green%' ) as profit group by nation, o_year order by nation, o_year desc;

-- q10
select
            c_custkey,
            c_name,
            sum(l_extendedprice * (1 - l_discount)) as revenue,
            c_acctbal,
            n_name,
            c_address,
            c_phone,
            c_comment
        from
            customer,
            orders,
            lineitem,
            nation
        where
            c_custkey = o_custkey
            and l_orderkey = o_orderkey
            and o_orderdate >= date '1993-10-01'
            and o_orderdate < date '1993-10-01' + interval '3' month
            and l_returnflag = 'R'
            and c_nationkey = n_nationkey
        group by
            c_custkey,
            c_name,
            c_acctbal,
            c_phone,
            n_name,
            c_address,
            c_comment
        order by
            revenue desc
        limit 20;

-- q10_duckdb_join_order
select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue, c_acctbal, n_name, c_address, c_phone, c_comment from (customer join nation on c_nationkey = n_nationkey) join (lineitem join orders on l_orderkey = o_orderkey) on c_custkey = o_custkey where o_orderdate >= date '1993-10-01' and o_orderdate < date '1993-10-01' + interval '3' month and l_returnflag = 'R' group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by revenue desc limit 20;

-- q11
select
            ps_partkey,
            sum(ps_supplycost * ps_availqty) as value
        from
            partsupp,
            supplier,
            nation
        where
            ps_suppkey = s_suppkey
            and s_nationkey = n_nationkey
            and n_name = 'GERMANY'
        group by
            ps_partkey having
                sum(ps_supplycost * ps_availqty) > (
                    select
                        sum(ps_supplycost * ps_availqty) * 0.0001
                    from
                        partsupp,
                        supplier,
                        nation
                    where
                        ps_suppkey = s_suppkey
                        and s_nationkey = n_nationkey
                        and n_name = 'GERMANY'
                )
        order by
            value desc;

-- q11_duckdb_join_order
select ps_partkey, sum(ps_supplycost * ps_availqty) as value from partsupp join (supplier join nation on s_nationkey = n_nationkey) on ps_suppkey = s_suppkey where n_name = 'GERMANY' group by ps_partkey having sum(ps_supplycost * ps_availqty) > ( select sum(ps_supplycost * ps_availqty) * 0.0001 from partsupp join (supplier join nation on s_nationkey = n_nationkey) on ps_suppkey = s_suppkey where n_name = 'GERMANY' ) order by value desc;

-- q18
select
            c_name,
            c_custkey,
            o_orderkey,
            o_orderdate,
            o_totalprice,
            sum(l_quantity)
        from
            customer,
            orders,
            lineitem
        where
            o_orderkey in (
                select
                    l_orderkey
                from
                    lineitem
                group by
                    l_orderkey having
                        sum(l_quantity) > 300
            )
            and c_custkey = o_custkey
            and o_orderkey = l_orderkey
        group by
            c_name,
            c_custkey,
            o_orderkey,
            o_orderdate,
            o_totalprice
        order by
            o_totalprice desc,
            o_orderdate
        limit 100;

-- q18_duckdb_join_order
select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity) from lineitem join ((select * from orders where o_orderkey in ( select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > 300 )) o join customer on o_custkey = c_custkey) on l_orderkey = o_orderkey group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice order by o_totalprice desc, o_orderdate limit 100;

-- q21
select
            s_name,
            count(*) as numwait
        from
            supplier,
            lineitem l1,
            orders,
            nation
        where
            s_suppkey = l1.l_suppkey
            and o_orderkey = l1.l_orderkey
            and o_orderstatus = 'F'
            and l1.l_receiptdate > l1.l_commitdate
            and exists (
                select
                    *
                from
                    lineitem l2
                where
                    l2.l_orderkey = l1.l_orderkey
                    and l2.l_suppkey <> l1.l_suppkey
            )
            and not exists (
                select
                    *
                from
                    lineitem l3
                where
                    l3.l_orderkey = l1.l_orderkey
                    and l3.l_suppkey <> l1.l_suppkey
                    and l3.l_receiptdate > l3.l_commitdate
            )
            and s_nationkey = n_nationkey
            and n_name = 'SAUDI ARABIA'
        group by
            s_name
        order by
            numwait desc,
            s_name
        limit 100;

-- q21_duckdb_join_order
select s_name, count(*) as numwait from ( select s_name, l1.l_orderkey as l_orderkey, l1.l_suppkey as l_suppkey from (lineitem l1 join (supplier join nation on s_nationkey = n_nationkey) on s_suppkey = l1.l_suppkey) join orders on o_orderkey = l1.l_orderkey where o_orderstatus = 'F' and l1.l_receiptdate > l1.l_commitdate and n_name = 'SAUDI ARABIA' and not exists ( select * from lineitem l3 where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey and l3.l_receiptdate > l3.l_commitdate ) ) t where exists ( select * from lineitem l2 where l2.l_orderkey = t.l_orderkey and l2.l_suppkey <> t.l_suppkey ) group by s_name order by numwait desc, s_name limit 100;

-- q15
with revenue0 (supplier_no, total_revenue) as (
            select
                l_suppkey,
                sum(l_extendedprice * (1 - l_discount))
            from
                lineitem
            where
                l_shipdate >= date '1996-01-01'
                and l_shipdate < date '1996-01-01' + interval '3' month
            group by
                l_suppkey
        )
        select
            s_suppkey,
            s_name,
            s_address,
            s_phone,
            total_revenue
        from
            supplier,
            revenue0
        where
            s_suppkey = supplier_no
            and total_revenue = (
                select
                    max(total_revenue)
                from
                    revenue0
            )
        order by
            s_suppkey;

-- q15_shared_cte
with revenue0 as materialized ( select l_suppkey as supplier_no, sum(l_extendedprice * (1 - l_discount)) as total_revenue from lineitem where l_shipdate >= date '1996-01-01' and l_shipdate < date '1996-01-01' + interval '3' month group by l_suppkey ) select s_suppkey, s_name, s_address, s_phone, total_revenue from supplier, revenue0 where s_suppkey = supplier_no and total_revenue = ( select max(total_revenue) from revenue0 ) order by s_suppkey;

-- hits23
SELECT * FROM hits WHERE "URL" LIKE '%google%'
               ORDER BY "EventTime", "WatchID" LIMIT 10;

-- hits23_sort_keys_only
SELECT "WatchID", "EventTime" FROM hits WHERE "URL" LIKE '%google%' ORDER BY "EventTime", "WatchID" LIMIT 10;
