object UnderstandMR:
  def payload(i:Int):List[Int] = List(i-1,i,i+1)

  def mymap(data: List[Int])(f: Int => List[Int]): List[List[Int]] =
    data match
      case hd::tl => f(hd) :: mymap(tl)(f)
      case List() => List()

  @main def runMR =
    val cod = List(1,3,5,6,8)//List(List(0,1,2),List(2,3,4),....
    val result = cod.map(payload)
    val myresult = mymap(cod)(payload)

    println(result)
    println(myresult)