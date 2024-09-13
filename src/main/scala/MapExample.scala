object MapExample:
  def f(s: String): Int = s.length

  def main(args: Array[String]): Unit = {
    val lst = List("abc", "defg", "k")
    println(lst.map(f))
  }