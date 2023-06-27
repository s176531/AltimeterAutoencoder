import numpy as np
import pickle
from typing import List, Tuple, BinaryIO, Dict, Any, Type, Literal


class Regression:
    def __init__(
            self,
            fit_type: Tuple[str, ...] | str = "poly",
            deg: Tuple[int, ...] | int = 1,
            period: int = 1
        ):
        if isinstance(fit_type, str) and isinstance(deg, tuple):
            raise ValueError(f"Multiple degrees, but only one type given.")
        elif isinstance(fit_type, tuple) and isinstance(deg, tuple):
            if len(fit_type) != len(deg):
                raise ValueError(f"Number of types and degrees must match.")
        
        # type(s) of regression
        self.fit_type = fit_type

        # degree for (each) type of regression
        self.deg = deg

        # period of harmonic fits. P=1 means deg=1 is yearly period
        self.period = period
        self.coefs = None
        self._trained = False

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        return self._trained

    @staticmethod
    def coef_length(fit_type: Tuple[str, ...] | str, deg: Tuple[int, ...] | int):
        """Get number of coefficients"""
        # Error handling
        if isinstance(fit_type, str) and isinstance(deg, tuple):
            raise ValueError(f"Multiple degrees, but only one type given.")
        elif isinstance(fit_type, tuple) and isinstance(deg, tuple):
            if len(fit_type) != len(deg):
                raise ValueError(f"Number of types and degrees must match.")
        
        length = 1
        if isinstance(deg, int) and isinstance(fit_type, str):
            mult = 2 if fit_type == "fourier" else 1
            length += deg * mult
        elif isinstance(deg, int) and isinstance(fit_type, tuple):
            for t in fit_type:
                length += Regression.coef_length(t, deg) - 1
        elif isinstance(deg, tuple) and isinstance(fit_type, tuple):
            for d, t in zip(deg, fit_type):
                length += Regression.coef_length(t, d) - 1
        return length

    def create_kernel(self, time):
        """Create kernel matrix"""
        if isinstance(self.deg, int):
            if self.deg < 0:
                raise ValueError(f"Expected degree greater than or equal to 0, got {self.deg}")
        if isinstance(self.deg, tuple):
            for deg in self.deg:
                if deg < 0:
                    raise ValueError(f"Expected all degrees to be greater than or equal to 0, got {self.deg}")
        kernel = np.ones(time.shape)
        kernel_elements = self.find_kernel_structure()
        for element in kernel_elements:
            kernel = self.add_kernel_element(kernel, time, *element)
        return kernel

    def find_kernel_structure(self) -> List[Tuple[str,int]]:
        """Determine kernel structure from input fit type and degree"""
        if isinstance(self.fit_type, str) and isinstance(self.deg, int):
            kernel_elements = [(self.fit_type, deg) for deg in range(1, self.deg+1)]
        elif isinstance(self.fit_type, tuple) and isinstance(self.deg, int):
            kernel_elements = [(ft, deg) for deg in range(1 ,self.deg+1) for ft in self.fit_type]
        elif isinstance(self.fit_type, str) and isinstance(self.deg, tuple):
            raise ValueError(f"Multiple degrees, but only one type given. Lengths must match")
        elif isinstance(self.fit_type, tuple) and isinstance(self.deg, tuple):
            kernel_elements = []
            for ft,deg in zip(self.fit_type, self.deg):
                kernel_elements.extend([(ft, d) for d in range(1,deg+1)])
        else:
            raise ValueError(f"Expected fit_type to be of type int or tuple, got {type(self.fit_type)}")
        return kernel_elements

    def add_kernel_element(self, kernel, time, element: str, i: int):
        """Constructor for adding elements to kernel matrix"""
        if element == "poly":
            kernel_out = np.hstack([kernel, time**i])
        elif element == "sin":
            kernel_out = np.hstack([kernel, np.sin(i*2*np.pi*time / self.period)])
        elif element == "cos":
            kernel_out = np.hstack([kernel, np.cos(i*2*np.pi*time / self.period)])
        elif element == "fourier":
            kernel_out = np.hstack([kernel, np.sin(i*2*np.pi*time / self.period), np.cos(i*2*np.pi*time / self.period)])
        else:
            raise ValueError('Use one of fit_types: "poly", "sin", "cos" or "fourier"')
        return kernel_out # type: ignore
        
    def fit(self, x, y) -> None:
        """
        Get model coefficients
        x: time
        y: feature(s)
        """
        rm_nan_idx = ~np.isnan(y)

        # Converted time from ns to yr
        time = x[rm_nan_idx].astype(np.int64).reshape(-1,1)/(365.25*24*3600e9)
        y = y[rm_nan_idx].reshape(-1, 1)
        kernel = self.create_kernel(time)
        if len(y) == 0:
            coefs = np.full(kernel.shape[1], np.nan)
        else:
            coefs = np.linalg.lstsq(kernel, y, rcond=None)[0]
        self.coefs = coefs.flatten()
        self._trained = True
    
    def predict(self, x):
        """
        Evaluate regression in input time values
        x: time
        """
        if not self._trained or self.coefs is None:
            raise ValueError("Fit must be called before predict")
        eval_time = x.astype(np.int64).reshape(-1,1)/(365.25*24*3600e9) # ns -> yr
        kernel = self.create_kernel(eval_time)
        return kernel@self.coefs
    
    def save(self, file: BinaryIO) -> None:
        """ Saves model to a file"""
        pickle.dump(self.get_parameters(), file)

    @classmethod
    def load(cls, file: BinaryIO):
        """ Loads the model from a file"""
        return cls.set_parameters(pickle.load(file))
    
    @classmethod
    def set_parameters(cls, parameters: Dict[str, Any]):
        """Reinitialize instance of class"""
        regression = cls(parameters.get("fit_type"), parameters.get("deg"), parameters.get("period")) # type: ignore
        regression.coefs = parameters.get("coefs")
        regression._trained = parameters.get("_trained")
        return regression
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters to reconstruct instance of class"""
        return {
            "fit_type": self.fit_type,
            "deg": self.deg,
            "period": self.period,
            "coefs": self.coefs,
            "_trained": self._trained
        }

class MetaRegression:

    def __init__(
            self,
            regressor,
            kwargs: Dict[str, Any],
            dim: Literal[0, 1]
        ):
        
       self.regressor = regressor
       self.kwargs = kwargs
       self.dim = dim
       self.internal_parameters: List[Dict[str, Any]] = []
       self.valid_regressor_map: Dict[int, int] = {}
       self._trained = False
       self._y_shape: Tuple[int, ...] | None = None

    @property
    def IsTrained(self) -> bool:
        """
        True of the fit function has successfully
        fitted the model otherwise False.
        """
        return self._trained

    def dim_lookup(self, i: int) -> Tuple[slice, int] | Tuple[int, slice]:
        """"""
        if self.dim == 0:
            return (slice(None), i)
        elif self.dim == 1:
            return (i, slice(None))
        else:
            raise ValueError("Invalid dimension.")
    
    @property
    def invert_dim(self) -> Literal[0, 1]:
        return 1 - self.dim # type: ignore

    def fit(self, x, y):
        """
        Fitting x ~ Y where x is a vector that maps to each element in the matrix Y.
        This means x ~ Y[0], x ~ Y[1], ...
        """
        self.internal_parameters: List[Dict[str, Any]] = []
        valid_parameters = None
        self._y_shape = y.shape

        model_id = 0
        for i in range(self._y_shape[self.invert_dim]):
            
            # Get the correct dimension of y
            d = y[self.dim_lookup(i)]

            # Check d does not only contain nan
            if np.isnan(d).all():
                continue
            
            # Setup regressor and fit model
            regressor = self.regressor(**self.kwargs) # type: ignore
            regressor.fit(x, d)

            # Get model parameters
            parameters = regressor.get_parameters()
            
            # Get parameters what is not also saved in kwargs
            if valid_parameters is None:
                valid_parameters = set(self.kwargs) ^ set(parameters)

            # Save parameters of regressor
            self.internal_parameters.append({valid_parameter: parameters[valid_parameter] for valid_parameter in valid_parameters})

            # Map fitting id to regressor id
            self.valid_regressor_map[i] = model_id
            model_id += 1
        self._trained = True

    def predict(self, x):
        """ Makes a prediction using x: time"""
        if not self._trained or self._y_shape is None:
            raise ValueError("Fit must be called before predict")

        predictions = np.full((len(x), self._y_shape[self.invert_dim]), np.nan)
        for i in range(self._y_shape[self.invert_dim]):
            # Get regressor mapping
            if (idx := self.valid_regressor_map.get(i)) is None:
                continue
            
            # Construct regressor
            regressor = self.regressor.set_parameters(self.kwargs | self.internal_parameters[idx])
            
            # Predict using the regressor
            predictions[self.dim_lookup(i)] = regressor.predict(x)
        return predictions

    def save(self, file: BinaryIO) -> None:
        """ Saves model to a file"""
        pickle.dump(self.get_parameters(), file)

    @classmethod
    def load(cls, file: BinaryIO):
        """ Loads the model from a file"""
        return cls.set_parameters(pickle.load(file))
    
    @classmethod
    def set_parameters(cls, parameters: Dict[str, Any]):
        """Reinitialize instance of class"""
        regressor = cls(parameters.get("regressor"), parameters.get("kwargs"), parameters.get("dim")) # type: ignore
        regressor.internal_parameters = parameters["internal_parameters"]
        regressor.valid_regressor_map = parameters["valid_regressor_map"]
        regressor._trained = parameters["_trained"]
        regressor._y_shape = parameters["_y_shape"]
        return regressor
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return parameters to reconstruct instance of class"""
        return {
            "regressor": self.regressor,
            "kwargs": self.kwargs,
            "dim": self.dim,
            "internal_parameters": self.internal_parameters,
            "valid_regressor_map": self.valid_regressor_map,
            "_trained": self._trained,
            "_y_shape": self._y_shape
        }
