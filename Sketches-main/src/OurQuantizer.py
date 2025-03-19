import numpy as np


# פונקציה לקוונטיזציה סימטרית
def quantize_symmetric(values, step_size):
    """
    פונקציה שמבצעת קוונטיזציה סימטרית על הערכים שניתנים לה.

    :param values: מערך של ערכים (וקטור או גריד) שנרצה לקוונטז.
    :param step_size: גודל הצעד לקוונטיזציה (המרחק בין הערכים המוכרים).
    :return: מערך של ערכים מקוונטזים.
    """
    # מחלקים את הערכים בגודל הצעד, מעגלים למספר שלם קרוב, ואז מכפילים בחזרה בגודל הצעד.
    # בכך אנחנו משאירים את הערכים על הערכים המקובלים בתהליך הקוונטיזציה.
    quantized_values = np.round(values / step_size) * step_size

    return quantized_values


# פונקציה לדה-קוונטיזציה סימטרית
def dequantize_symmetric(quantized_values, step_size):
    """
    פונקציה שמבצעת דה-קוונטיזציה סימטרית על הערכים המקוונטזים שניתנים לה.

    :param quantized_values: מערך של ערכים מקוונטזים שנרצה להחזיר לפורמט המקורי.
    :param step_size: גודל הצעד של הקוונטיזציה.
    :return: מערך של ערכים מדה-קוונטזים.
    """
    # בדה-קוונטיזציה, לא צריך לבצע שום שינוי. הערכים נשארים כפי שהם,
    # כי המטרה היא להחזיר אותם לרמת הדיוק שלהם לאחר הקוונטיזציה.
    return quantized_values


# דוגמת שימוש
values = np.array([1.2, -3.5, 0.7, 4.8, -2.2])  # וקטור לדוגמה עם ערכים חיוביים ושליליים
step_size = 1.0  # גודל הצעד לקוונטיזציה (למשל, כל מספר יעוגל למספר הקרוב ביותר שווה בערך ל-1)

# מבצע קוונטיזציה על הערכים
quantized_values = quantize_symmetric(values, step_size)
print("Quantized Values:", quantized_values)

# מבצע דה-קוונטיזציה (במקרה הזה אין שינוי)
dequantized_values = dequantize_symmetric(quantized_values, step_size)
print("Dequantized Values:", dequantized_values)
