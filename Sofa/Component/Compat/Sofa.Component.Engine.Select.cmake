set(SOFAENGINE_SRC src/SofaEngine)
set(SOFAGENERALENGINE_SRC src/SofaGeneralEngine)

list(APPEND HEADER_FILES
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.h
    ${SOFAGENERALENGINE_SRC}/IndicesFromValues.inl
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.h
    ${SOFAGENERALENGINE_SRC}/PointsFromIndices.inl
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromIndices.inl
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.h
    ${SOFAGENERALENGINE_SRC}/ValuesFromPositions.inl
    ${SOFAGENERALENGINE_SRC}/MeshSampler.h
    ${SOFAGENERALENGINE_SRC}/MeshSampler.inl
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshSplittingEngine.inl
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.h
    ${SOFAGENERALENGINE_SRC}/MeshSubsetEngine.inl
    ${SOFAENGINE_SRC}/BoxROI.h
    ${SOFAENGINE_SRC}/BoxROI.inl
    ${SOFAGENERALENGINE_SRC}/ComplementaryROI.h
    ${SOFAGENERALENGINE_SRC}/ComplementaryROI.inl
    ${SOFAGENERALENGINE_SRC}/MergeROIs.h
    ${SOFAGENERALENGINE_SRC}/MeshBoundaryROI.h
    ${SOFAGENERALENGINE_SRC}/MeshROI.h
    ${SOFAGENERALENGINE_SRC}/MeshROI.inl
    ${SOFAGENERALENGINE_SRC}/NearestPointROI.h
    ${SOFAGENERALENGINE_SRC}/NearestPointROI.inl
    ${SOFAGENERALENGINE_SRC}/PairBoxRoi.h
    ${SOFAGENERALENGINE_SRC}/PairBoxRoi.inl
    ${SOFAGENERALENGINE_SRC}/PlaneROI.h
    ${SOFAGENERALENGINE_SRC}/PlaneROI.inl
    ${SOFAGENERALENGINE_SRC}/ProximityROI.h
    ${SOFAGENERALENGINE_SRC}/ProximityROI.inl
    ${SOFAGENERALENGINE_SRC}/SelectLabelROI.h
    ${SOFAGENERALENGINE_SRC}/SelectConnectedLabelsROI.h
    ${SOFAGENERALENGINE_SRC}/SphereROI.h
    ${SOFAGENERALENGINE_SRC}/SphereROI.inl
    ${SOFAGENERALENGINE_SRC}/SubsetTopology.h
    ${SOFAGENERALENGINE_SRC}/SubsetTopology.inl
)