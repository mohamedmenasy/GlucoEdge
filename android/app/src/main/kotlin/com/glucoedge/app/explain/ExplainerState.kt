package com.glucoedge.app.explain

/** UI state of the explanation feature. Hidden means: no model file on device. */
sealed interface ExplainerState {
    data object Hidden : ExplainerState
    data object Ready : ExplainerState
    data object LoadingModel : ExplainerState
    data object Generating : ExplainerState
    data class Note(val text: String) : ExplainerState
    data class Error(val message: String) : ExplainerState
}
