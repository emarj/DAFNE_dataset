from typing import Tuple

class VersionType:
    def __init__(self, version: str, type_: str):

        self.version = Version(version)
        
        if not isinstance(type_, str) or not type_.strip():
            raise ValueError("Type must be a non-empty string")
        
        self.type_ = type_

    @classmethod
    def from_str(cls, version_type_str: str):
        try:
            name, type_str = version_type_str.rsplit('_', 1)
        except ValueError:
            raise ValueError(f"Invalid version_type string: {version_type_str}. Expected format: '<version>_<type>'")
        
        return cls(name, type_str)

    def __str__(self):
        return f"{self.type_}_{self.version}"
    
    def __repr__(self):
        return f"VersionType(version={self.version}, type_={self.type_})"
    
    def major_version_type(self) -> str:
        return f"{self.type_}_{self.version.major_str()}"

    def supported(self, supported_versions_types: dict) -> bool:
        return self.type_ in supported_versions_types and str(self.version) in supported_versions_types[self.type_]
    

class Version:
    def __init__(self, version_str: str):
        self.parts, self.suffix, valid = validate_version_str(version_str)
        if not valid:
            raise ValueError(f"Version must be a string in the format v0.0.0a. Invalid version string: {version_str}")

    def __str__(self):
        # omit trailing zero parts for a compact representation, keep at least the major
        parts = self.parts[:]
        while len(parts) > 1 and parts[-1] == 0:
            parts.pop()
        return f"v{'.'.join(str(p) for p in parts)}{self.suffix}"
    
    def full_str(self) -> str:
        return f"v{'.'.join(str(p) for p in self.parts)}{self.suffix}"
    
    def __repr__(self):
        return f"Version({self.__str__()})"
    
    def major_str(self) -> str:
        return f"v{self.parts[0]}"
    
    def major(self) -> int:
        return self.parts[0]
    
    def minor(self) -> int:
        return self.parts[1]
    
    def patch(self) -> int:
        return self.parts[2]
    
def validate_version_str(version_str: str) -> Tuple[list, str, bool]:
    if not isinstance(version_str, str):
        return [], '', False
    if not version_str.startswith('v'):
        return [], '', False
    body = version_str[1:]
    if not body:
        return [], '', False

    suffix = ''
    if body[-1].isalpha():
        suffix = body[-1]
        body = body[:-1]
        if not body:
            return [], '', False

    parts = body.split('.')
    if len(parts) > 3 or not all(p.isdigit() for p in parts):
        return [], '', False
    
    parts_int = [int(p) for p in parts]
    
    # pad parts with '0' until length is 3
    while len(parts_int) < 3:
        parts_int.append(0)
    
    return parts_int, suffix, True