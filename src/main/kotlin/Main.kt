import org.jetbrains.numkt.*
import org.jetbrains.numkt.core.*
import org.jetbrains.numkt.math.*

@ExperimentalNumkt
fun main() {
    val a = arange(15).reshape(3, 5) // KtNDArray<Int>([[ 0,  1,  2,  3,  4],
    // [ 5,  6,  7,  8,  9],
    // [10, 11, 12, 13, 14]]

    println(a.shape.contentEquals(intArrayOf(3, 5))) // true
    println(a.ndim == 2) // true
    println(a.dtype) // class java.lang.Integer

    // create an array of ints, we square each element and the shape to (3, 5)
    val b = (arange(15) `**` 2).reshape(3, 5)

    // c is the product of a and b, element-wise
    val c = a * b
    println(c)
    // Output:
    // [[   0    1    8   27   64]
    //  [ 125  216  343  512  729]
    //  [1000 1331 1728 2197 2744]]

    // d is the dot product of the transposed c and a
    val d = c.transpose().dot(a)
    println(d)
    // Output:
    // [[10625 11750 12875 14000 15125]
    //  [14390 15938 17486 19034 20582]
    //  [18995 21074 23153 25232 27311]
    //  [24530 27266 30002 32738 35474]
    //  [31085 34622 38159 41696 45233]]
}
