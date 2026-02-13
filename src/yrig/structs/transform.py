import math


class Vector3:
    """A minimal 3D vector with basic arithmetic and length operations.

    Supports component-wise addition, subtraction, multiplication, and division
    with other ``Vector3`` instances or scalar values (``float`` / ``int``).

    Attributes:
        x: The X component of the vector.
        y: The Y component of the vector.
        z: The Z component of the vector.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        """Initialize a Vector3.

        Args:
            x: The X component. Defaults to ``0``.
            y: The Y component. Defaults to ``0``.
            z: The Z component. Defaults to ``0``.
        """
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        """Return a human-readable string representation ``(x, y, z)``."""
        return f"({self.x},{self.y},{self.z})"

    def __repr__(self):
        """Return a string representation suitable for debugging."""
        return f"({self.x},{self.y},{self.z})"

    def __add__(self, other):
        """Add another ``Vector3`` or a scalar to this vector (component-wise).

        Args:
            other: A ``Vector3`` for component-wise addition, or a scalar
                (``float`` / ``int``) that is added to every component.

        Returns:
            A new ``Vector3`` with the summed components, or ``self``
            unchanged if *other* is an unsupported type.
        """
        if type(other) is Vector3:
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif type(other) is float or int:
            return Vector3(self.x + other, self.y + other, self.z + other)
        else:
            return self

    def __radd__(self, other):
        """Support ``scalar + Vector3`` by delegating to `__add__`.

        Args:
            other: A scalar (``float`` / ``int``) to add to every component.

        Returns:
            A new ``Vector3`` with the summed components, or ``self``
            unchanged if *other* is an unsupported type.
        """
        if type(other) is float or int:
            return Vector3(self.x + other, self.y + other, self.z + other)
        else:
            return self

    def __sub__(self, other):
        """Subtract another ``Vector3`` from this vector (component-wise).

        Args:
            other: A ``Vector3`` whose components are subtracted from this
                vector's components.

        Returns:
            A new ``Vector3`` with the difference, or ``self`` unchanged
            if *other* is not a ``Vector3``.
        """
        if type(other) is Vector3:
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return self

    def __mul__(self, other):
        """Multiply by another ``Vector3`` (component-wise) or a scalar.

        Args:
            other: A ``Vector3`` for component-wise multiplication, or a
                scalar (``float`` / ``int``) to scale every component.

        Returns:
            A new ``Vector3`` with the product, or ``self`` unchanged if
            *other* is an unsupported type.
        """
        if type(other) is Vector3:
            return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif type(other) is float or int:
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            return self

    def __rmul__(self, other):
        """Support ``scalar * Vector3`` by delegating to `__mul__`.

        Args:
            other: A scalar (``float`` / ``int``) to scale every component.

        Returns:
            A new ``Vector3`` with the scaled components, or ``self``
            unchanged if *other* is an unsupported type.
        """
        if type(other) is float or int:
            return Vector3(self.x * other, self.y * other, self.z * other)
        else:
            return self

    def __truediv__(self, other):
        """Divide by another ``Vector3`` (component-wise) or a scalar.

        Args:
            other: A ``Vector3`` for component-wise division, or a scalar
                (``float`` / ``int``) to divide every component by.

        Returns:
            A new ``Vector3`` with the quotient, or ``self`` unchanged if
            *other* is an unsupported type.

        Raises:
            ZeroDivisionError: If any divisor component (or the scalar) is zero.
        """
        if type(other) is Vector3:
            return Vector3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif type(other) is float or int:
            return Vector3(self.x / other, self.y / other, self.z / other)
        else:
            return self

    def length(self) -> float:
        """Return the Euclidean length (magnitude) of the vector.

        Returns:
            The non-negative length computed as ``sqrt(x² + y² + z²)``.
        """
        return abs(math.sqrt((self.x**2) + (self.y**2) + (self.z**2)))
