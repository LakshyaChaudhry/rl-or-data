# Pool sample — 20 problems, seed 20260908

- Pool: `data/pool/pool.jsonl`
- Pool seed: 20260901  ·  config hash: `4af3d8eac9b1db4ff54403d7deadd739dd61edcdfeb37fdf1d805b4dd72ad43a`
- Generated at: 2026-09-07T23:06:12+00:00  ·  git SHA: `dc344a5c518724b28779c0099ec77fa8a909f76d`
- Cell counts: S2=750, S3=750, S4=750, S5=750, M2=750, M3=750, M4=750, M5=750
- Invariants checked on every sampled problem: no no-op step, no empty step, ≥3 values at the final op.

## 1. `791c48a8d3a7` — S, 2 steps

> Start with the inclusive integer range [7, 33]. First, keep only the numbers whose digits sum to 6. Of these numbers, give the largest remaining number.

- **Answer:** 33
- **Set sizes:** range: 27 → digit_sum_equals(s=6): 3
- Pipeline: `{"filters":[{"name":"digit_sum_equals","s":6}],"op":{"name":"max"},"range":{"hi":33,"lo":7},"transforms":[]}`

## 2. `d4f5a92198fe` — S, 3 steps

> Consider the integers from 1 to 50, inclusive. First, keep only the numbers that are divisible by 5. Then, add 7 to each remaining number. Finally, report the non-negative remainder when the product of the remaining numbers is divided by 82.

- **Answer:** 66
- **Set sizes:** range: 50 → divisible_by(n=5): 10 → add(k=7): 10
- Pipeline: `{"filters":[{"n":5,"name":"divisible_by"}],"op":{"m":82,"name":"product_mod"},"range":{"hi":50,"lo":1},"transforms":[{"k":7,"name":"add"}]}`

## 3. `8ba9b9766784` — S, 3 steps

> Start with the inclusive integer range [1, 49]. First, retain only the values below 29 (not including 29 itself). Next, retain only the multiples of 5. Finally, report the result of combining all remaining numbers with bitwise AND.

- **Answer:** 0
- **Set sizes:** range: 49 → below_threshold(t=29): 28 → divisible_by(n=5): 5
- Pipeline: `{"filters":[{"name":"below_threshold","t":29},{"n":5,"name":"divisible_by"}],"op":{"name":"bitwise_and"},"range":{"hi":49,"lo":1},"transforms":[]}`

## 4. `4ca793f9d7c5` — S, 3 steps

> Consider the integers from 4 to 36, inclusive. First, retain only the multiples of 8. Then, square each remaining number. Of these numbers, report how many numbers are left.

- **Answer:** 4
- **Set sizes:** range: 33 → divisible_by(n=8): 4 → square: 4
- Pipeline: `{"filters":[{"n":8,"name":"divisible_by"}],"op":{"name":"count"},"range":{"hi":36,"lo":4},"transforms":[{"name":"square"}]}`

## 5. `25d95a0a5ac2` — S, 3 steps

> Consider the integers from 2 to 35, inclusive. First, discard every number that does not have a 2 among its digits. Next, discard every even number, keeping the odd ones. Of these numbers, report the result of combining all remaining numbers with bitwise AND.

- **Answer:** 17
- **Set sizes:** range: 34 → contains_digit(d=2): 13 → odd: 5
- Pipeline: `{"filters":[{"d":2,"name":"contains_digit"},{"name":"odd"}],"op":{"name":"bitwise_and"},"range":{"hi":35,"lo":2},"transforms":[]}`

## 6. `8e5b855b3af1` — S, 4 steps

> Take all integers between 15 and 45, including both endpoints. First, discard every multiple of 12. Next, reverse the decimal digits of each remaining number (reverse the digits of its absolute value, keep the original sign, and drop any leading zeros). Then, replace each remaining number with the sum of its decimal digits (ignoring any minus sign). Finally, compute the bitwise XOR of all remaining values.

- **Answer:** 10
- **Set sizes:** range: 31 → not_divisible_by(n=12): 29 → reverse_digits: 29 → digit_sum: 29
- Pipeline: `{"filters":[{"n":12,"name":"not_divisible_by"}],"op":{"name":"bitwise_xor"},"range":{"hi":45,"lo":15},"transforms":[{"name":"reverse_digits"},{"name":"digit_sum"}]}`

## 7. `d11e1b862047` — S, 4 steps

> Consider the integers from 9 to 37, inclusive. First, discard every number that is not a perfect square. Then, add 3 to each remaining number. Next, take every remaining value modulo 15, using the non-negative remainder (0 to 14). Of these numbers, compute the mean of the remaining values, rounded to the nearest integer (round half to even).

- **Answer:** 10
- **Set sizes:** range: 29 → perfect_square: 4 → add(k=3): 4 → modulo(m=15): 4
- Pipeline: `{"filters":[{"name":"perfect_square"}],"op":{"name":"mean"},"range":{"hi":37,"lo":9},"transforms":[{"k":3,"name":"add"},{"m":15,"name":"modulo"}]}`

## 8. `30e42c6784d3` — S, 5 steps

> Consider the integers from 10 to 24, inclusive. First, keep only the prime numbers. Then, replace each remaining number with the sum of its decimal digits (ignoring any minus sign). Next, replace each remaining number x with 4 * x. Next, map every remaining value to the digit sum of its absolute value. Of these numbers, give the difference between the largest and the smallest remaining number.

- **Answer:** 6
- **Set sizes:** range: 15 → prime: 5 → digit_sum: 5 → multiply(k=4): 5 → digit_sum: 5
- Pipeline: `{"filters":[{"name":"prime"}],"op":{"name":"range"},"range":{"hi":24,"lo":10},"transforms":[{"name":"digit_sum"},{"k":4,"name":"multiply"},{"name":"digit_sum"}]}`

## 9. `4e468352f26f` — S, 5 steps

> Take all integers between 9 and 51, including both endpoints. First, discard every number that is 21 or smaller. Then, retain only the odd integers. Then, replace each remaining number x with 2 * x. Then, decrease every remaining value by 8. Of these numbers, compute the product of all remaining values modulo 72 (the non-negative remainder).

- **Answer:** 0
- **Set sizes:** range: 43 → above_threshold(t=21): 30 → odd: 15 → multiply(k=2): 15 → add(k=-8): 15
- Pipeline: `{"filters":[{"name":"above_threshold","t":21},{"name":"odd"}],"op":{"m":72,"name":"product_mod"},"range":{"hi":51,"lo":9},"transforms":[{"k":2,"name":"multiply"},{"k":-8,"name":"add"}]}`

## 10. `c1eef7f6baa9` — M, 2 steps

> Take all integers between 59 and 197, including both endpoints. First, keep only the numbers that contain the digit 6. Of these numbers, report the average of the remaining numbers rounded to an integer, rounding halves to the even neighbour.

- **Answer:** 122
- **Set sizes:** range: 139 → contains_digit(d=6): 32
- Pipeline: `{"filters":[{"d":6,"name":"contains_digit"}],"op":{"name":"mean"},"range":{"hi":197,"lo":59},"transforms":[]}`

## 11. `8f7a32b66c23` — M, 2 steps

> Take all integers between 81 and 154, including both endpoints. First, keep only the prime numbers. Of these numbers, give the integer-rounded mean of the remaining values, using round-half-to-even.

- **Answer:** 117
- **Set sizes:** range: 74 → prime: 14
- Pipeline: `{"filters":[{"name":"prime"}],"op":{"name":"mean"},"range":{"hi":154,"lo":81},"transforms":[]}`

## 12. `7dcf8b3d63f3` — M, 3 steps

> Consider the integers from 14 to 198, inclusive. First, retain only the multiples of 4. Then, multiply each remaining number by 3. Finally, fold bitwise XOR across the remaining values and give the result.

- **Answer:** 892
- **Set sizes:** range: 185 → divisible_by(n=4): 46 → multiply(k=3): 46
- Pipeline: `{"filters":[{"n":4,"name":"divisible_by"}],"op":{"name":"bitwise_xor"},"range":{"hi":198,"lo":14},"transforms":[{"k":3,"name":"multiply"}]}`

## 13. `20b07cc8b7c2` — M, 3 steps

> Take all integers between 11 and 201, including both endpoints. First, retain only the primes. Next, replace every remaining value x with the digits of |x| written in reverse order, with the original sign restored and leading zeros dropped. Of these numbers, compute the median of the remaining values (if the count is even, use the lower of the two middle values).

- **Answer:** 98
- **Set sizes:** range: 191 → prime: 42 → reverse_digits: 42
- Pipeline: `{"filters":[{"name":"prime"}],"op":{"name":"median"},"range":{"hi":201,"lo":11},"transforms":[{"name":"reverse_digits"}]}`

## 14. `fb6273517c22` — M, 3 steps

> Start with the inclusive integer range [29, 152]. First, keep only the numbers that are divisible by 5. Next, square each remaining number. Of these numbers, give the difference between the largest and the smallest remaining number.

- **Answer:** 21600
- **Set sizes:** range: 124 → divisible_by(n=5): 25 → square: 25
- Pipeline: `{"filters":[{"n":5,"name":"divisible_by"}],"op":{"name":"range"},"range":{"hi":152,"lo":29},"transforms":[{"name":"square"}]}`

## 15. `fd55be7bef40` — M, 3 steps

> Consider the integers from 15 to 139, inclusive. First, retain only the multiples of 11. Next, replace each remaining number with its remainder modulo 12, taken in the range 0 to 11. Finally, report the result of combining all remaining numbers with bitwise OR.

- **Answer:** 15
- **Set sizes:** range: 125 → divisible_by(n=11): 11 → modulo(m=12): 11
- Pipeline: `{"filters":[{"n":11,"name":"divisible_by"}],"op":{"name":"bitwise_or"},"range":{"hi":139,"lo":15},"transforms":[{"m":12,"name":"modulo"}]}`

## 16. `93518281a7c6` — M, 4 steps

> Start with the inclusive integer range [7, 194]. First, retain only the values that are perfect squares. Then, keep only the numbers that are strictly less than 127. Then, keep only the numbers that are strictly less than 117. Of these numbers, report the result of combining all remaining numbers with bitwise XOR.

- **Answer:** 96
- **Set sizes:** range: 188 → perfect_square: 11 → below_threshold(t=127): 9 → below_threshold(t=117): 8
- Pipeline: `{"filters":[{"name":"perfect_square"},{"name":"below_threshold","t":127},{"name":"below_threshold","t":117}],"op":{"name":"bitwise_xor"},"range":{"hi":194,"lo":7},"transforms":[]}`

## 17. `1d70a5ad0a8a` — M, 4 steps

> Consider the integers from 32 to 200, inclusive. First, keep only the numbers that are odd. Next, map every remaining value to the digit sum of its absolute value. Next, scale every remaining value by a factor of 3. Of these numbers, compute the mean of the remaining values, rounded to the nearest integer (round half to even).

- **Answer:** 32
- **Set sizes:** range: 169 → odd: 84 → digit_sum: 84 → multiply(k=3): 84
- Pipeline: `{"filters":[{"name":"odd"}],"op":{"name":"mean"},"range":{"hi":200,"lo":32},"transforms":[{"name":"digit_sum"},{"k":3,"name":"multiply"}]}`

## 18. `eb533a6c6ea6` — M, 5 steps

> Take all integers between 83 and 173, including both endpoints. First, discard every number that is not prime. Next, replace each remaining number with its remainder modulo 19, taken in the range 0 to 18. Then, replace every remaining value x with the digits of |x| written in reverse order, with the original sign restored and leading zeros dropped. Next, replace each remaining number x with x + 2. Finally, compute the mean of the remaining values, rounded to the nearest integer (round half to even).

- **Answer:** 31
- **Set sizes:** range: 91 → prime: 18 → modulo(m=19): 18 → reverse_digits: 18 → add(k=2): 18
- Pipeline: `{"filters":[{"name":"prime"}],"op":{"name":"mean"},"range":{"hi":173,"lo":83},"transforms":[{"m":19,"name":"modulo"},{"name":"reverse_digits"},{"k":2,"name":"add"}]}`

## 19. `b37af5888fd5` — M, 5 steps

> Take all integers between 21 and 201, including both endpoints. First, keep only the numbers that are strictly less than 178. Then, keep only the numbers that are not divisible by 2. Then, keep only the numbers that are strictly less than 68. Then, replace every remaining value x with the digits of |x| written in reverse order, with the original sign restored and leading zeros dropped. Of these numbers, compute the bitwise XOR of all remaining values.

- **Answer:** 64
- **Set sizes:** range: 181 → below_threshold(t=178): 157 → not_divisible_by(n=2): 79 → below_threshold(t=68): 24 → reverse_digits: 24
- Pipeline: `{"filters":[{"name":"below_threshold","t":178},{"n":2,"name":"not_divisible_by"},{"name":"below_threshold","t":68}],"op":{"name":"bitwise_xor"},"range":{"hi":201,"lo":21},"transforms":[{"name":"reverse_digits"}]}`

## 20. `c968edd9f22c` — M, 5 steps

> Consider the integers from 13 to 198, inclusive. First, keep only the numbers that contain the digit 7. Next, replace each remaining number x with x * x. Next, replace each remaining number x with 5 * x. Next, reverse the decimal digits of each remaining number (reverse the digits of its absolute value, keep the original sign, and drop any leading zeros). Finally, find the greatest value among the remaining numbers.

- **Answer:** 548471
- **Set sizes:** range: 186 → contains_digit(d=7): 37 → square: 37 → multiply(k=5): 37 → reverse_digits: 37
- Pipeline: `{"filters":[{"d":7,"name":"contains_digit"}],"op":{"name":"max"},"range":{"hi":198,"lo":13},"transforms":[{"name":"square"},{"k":5,"name":"multiply"},{"name":"reverse_digits"}]}`

