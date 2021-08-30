package org.clulab.torchscript.utils

object StringUtils {

  def before(string: String, index: Int, all: Boolean, keep: Boolean): String = {
    if (index < 0)
      if (all) string
      else ""
    else string.substring(0, index + (if (keep) 1 else 0))
  }

  def beforeFirst(string: String, char: Char, all: Boolean = true, keep: Boolean = false): String =
    before(string, string.indexOf(char), all, keep)
}
